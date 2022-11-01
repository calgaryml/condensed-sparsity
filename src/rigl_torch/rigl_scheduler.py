""" implementation of https://arxiv.org/abs/1911.11134 """

import numpy as np
import torch
import torch.distributed as dist
from typing import List, Optional
import logging
import wandb

from rigl_torch.utils.rigl_utils import (
    get_W,
    get_static_filters_to_ablate,
    get_names_and_W,
)
from rigl_torch.meters.layer_meter import LayerMeter


class IndexMaskHook:
    def __init__(self, layer, scheduler):
        self.layer = layer
        self.scheduler = scheduler
        self.dense_grad = None

    def __name__(self):
        return "IndexMaskHook"

    @torch.no_grad()
    def __call__(self, grad):
        mask = self.scheduler.backward_masks[self.layer]

        # only calculate dense_grads when necessary
        if self.scheduler.check_if_backward_hook_should_accumulate_grad():
            if self.dense_grad is None:
                # initialize as all 0s so we can do a rolling average
                self.dense_grad = torch.zeros_like(grad)
            self.dense_grad += grad / self.scheduler.grad_accumulation_n
        else:
            self.dense_grad = None

        return grad * mask


def _create_step_wrapper(scheduler, optimizer):
    _unwrapped_step = optimizer.step

    def _wrapped_step():
        _unwrapped_step()
        scheduler.reset_momentum()
        scheduler.apply_mask_to_weights()

    optimizer.step = _wrapped_step


class RigLScheduler:
    _implemented_sparsity_distributions: List[str] = ["uniform", "erk"]

    def __init__(
        self,
        model,
        optimizer,
        dense_allocation=1,
        T_end=None,
        sparsity_distribution="uniform",
        ignore_linear_layers=True,
        delta=100,
        alpha=0.3,
        static_topo=False,
        grad_accumulation_n=1,
        state_dict=None,
        erk_power_scale=1.0,
        filter_ablation_threshold: Optional[float] = None,
        static_ablation: bool = True,
        dynamic_ablation: bool = False,
    ):
        self._logger = logging.getLogger(__file__)
        self.explored_params = None
        self.static_ablation = static_ablation
        self.dynamic_ablation = dynamic_ablation
        self.filter_ablation_threshold = filter_ablation_threshold
        self.erk_power_scale = erk_power_scale
        # define the actual schedule
        self.delta_T = delta
        self.alpha = alpha
        self.T_end = T_end
        self.dense_allocation = dense_allocation
        self.model = model
        self.optimizer = optimizer

        self.W, self._linear_layers_mask = get_W(
            model, return_linear_layers_mask=True
        )
        # modify optimizer.step() function to call "reset_momentum" after
        _create_step_wrapper(self, optimizer)

        self.N = [torch.numel(w) for w in self.W]

        if state_dict is not None:
            self.load_state_dict(state_dict)
            self.apply_mask_to_weights()

        else:
            self.sparsity_distribution = sparsity_distribution
            self.static_topo = static_topo
            self.grad_accumulation_n = grad_accumulation_n
            self.ignore_linear_layers = ignore_linear_layers
            self.backward_masks = None

            # define sparsity allocation
            self.S = self._allocate_sparsity()

            # determine if any filters must be ablated to meet filter-wise
            # sparsity threshold
            if self.static_ablation:
                self.static_ablated_filters = (
                    self.get_inital_num_filters_to_ablate()
                )
            else:
                self.static_ablated_filters = [0 for _ in range(len(self.W))]

            # randomly sparsify model according to S
            self.random_sparsify()

            # scheduler keeps a log of how many times it's called. this is how
            # it does its scheduling
            self.step = 0
            self.rigl_steps = 0
            self._update_itop_rs()

        # also, register backward hook so sparse elements cannot be recovered
        # during normal training
        self.backward_hook_objects = []
        for i, w in enumerate(self.W):
            # if sparsity is 0%, skip
            if self.S[i] <= 0:
                self.backward_hook_objects.append(None)
                continue

            if getattr(w, "_has_rigl_backward_hook", False):
                raise Exception(
                    "This model already has been registered to a RigLScheduler."
                )

            self.backward_hook_objects.append(IndexMaskHook(i, self))
            w.register_hook(self.backward_hook_objects[-1])
            setattr(w, "_has_rigl_backward_hook", True)
            self._register_meters()
            self._update_active_neurons()
            self._validate_params()

    def _validate_params(self) -> None:
        if self.static_ablation and self.dynamic_ablation:
            raise ValueError(
                "Only one of `static_ablation` and "
                "`dynamic ablation` may be True!"
            )
        self._validate_percent_params(self.dense_allocation, "dense_allocation")
        self._validate_percent_params(
            self.filter_ablation_threshold, "filter_ablation_threshold"
        )
        assert (
            self.grad_accumulation_n > 0
            and self.grad_accumulation_n < self.delta_T
        )
        assert (
            self.sparsity_distribution
            in self._implemented_sparsity_distributions
        )

    @torch.no_grad()
    def get_inital_num_filters_to_ablate(self) -> List[int]:
        """Populate list of filters to ablate with index of list corresponding
            to layer.

        Returns:
            List[int]: Layerwise filters to ablate.
        """
        inital_ablated_filters = []
        for idx, w in enumerate(self.W):
            # if sparsity is 0%, no neurons to ablate by defn.
            if (
                self.S[idx] <= 0
                or self.filter_ablation_threshold is None
                or self.filter_ablation_threshold <= 0
            ):
                inital_ablated_filters.append(0)
            else:
                inital_ablated_filters.append(
                    get_static_filters_to_ablate(
                        weight_tensor=w,
                        sparsity=self.S[idx],
                        filter_ablation_threshold=self.filter_ablation_threshold,  # noqa E501
                    )
                )
        return inital_ablated_filters

    def _allocate_sparsity(self) -> List[float]:
        sparsity_dist = []
        sparsity_allocators = {
            "uniform": self._uniform_sparsity_dist,
            "erk": self._erk_sparsity_dist,
        }
        if self.sparsity_distribution.lower() not in sparsity_allocators:
            raise ValueError(
                "Unknown sparsity distribution "
                f"{self.sparsity_distribution}. Please select from "
                f"{list(sparsity_allocators.keys())}."
            )
        sparsity_dist = sparsity_allocators[
            self.sparsity_distribution.lower()
        ]()
        return sparsity_dist

    def _uniform_sparsity_dist(self) -> List[float]:
        sparsity_dist = []
        for i, (W, is_linear) in enumerate(
            zip(self.W, self._linear_layers_mask)
        ):
            # when using uniform sparsity, the first layer is always 100%
            # dense UNLESS there is only 1 layer
            if i == 0 and len(self.W) > 1:
                sparsity_dist.append(0)

            elif is_linear and self.ignore_linear_layers:
                # if choosing to ignore linear layers, keep them 100% dense
                sparsity_dist.append(0)

            else:
                sparsity_dist.append(1 - self.dense_allocation)
        return sparsity_dist

    def _erk_sparsity_dist(self) -> List[float]:
        """Get Erdos Renyi Kernel sparsity distribution for `self.model`.

        Implementation based on approach in original rigl paper and reproduced
        papers:
        https://github.com/google-research/rigl/blob/97d62b0724c9a489a5318edb34951c6800575311/rigl/sparse_utils.py#L90
        https://github.com/varun19299/rigl-reproducibility/blob/f8a3398f6249e291aa8d91e376e49820fde8f2d3/sparselearning/funcs/init_scheme.py#L147


        Returns:
            List[float]: List of sparsities to apply per layer.
        """
        eps = None
        is_eps_valid = False
        dense_layers = set()
        while not is_eps_valid:
            divisor = 0
            rhs = 0
            raw_probabilties = {}
            for layer_idx, (weight_matrix, is_linear) in enumerate(
                zip(self.W, self._linear_layers_mask)
            ):
                n_params = np.prod(
                    weight_matrix.shape
                )  # Total number of params
                n_zeros = int(n_params * (1 - self.dense_allocation))
                n_ones = int(n_params * self.dense_allocation)

                if layer_idx in dense_layers or (
                    is_linear and self.ignore_linear_layers
                ):
                    # dense_layers.add(layer_idx)
                    rhs -= n_zeros
                else:
                    n_ones = n_params - n_zeros
                    rhs += n_ones
                    raw_prob = (
                        np.sum(weight_matrix.shape)
                        / np.prod(weight_matrix.shape)
                    ) ** self.erk_power_scale
                    raw_probabilties[layer_idx] = raw_prob
                    divisor += raw_probabilties[layer_idx] * n_params
            eps = rhs / divisor
            max_prob = np.max(list(raw_probabilties.values()))
            max_prob_eps = max_prob * eps
            if max_prob_eps > 1:
                is_eps_valid = False
                for layer_idx, raw_prob in raw_probabilties.items():
                    if raw_prob == max_prob:
                        self._logger.info(
                            f"Sparsity of layer at index {layer_idx} set to 0.0"
                        )
                        dense_layers.add(layer_idx)
                        break
            else:
                is_eps_valid = True

        sparsity_dist = []
        for layer_idx, (weight_matrix, is_linear) in enumerate(
            zip(self.W, self._linear_layers_mask)
        ):
            if layer_idx in dense_layers:
                sparsity = 0.0
            else:
                sparsity = 1 - (eps * raw_probabilties[layer_idx])
            sparsity_dist.append(sparsity)
        return sparsity_dist

    def _update_active_neurons(self) -> None:
        self.active_neurons = []
        for idx, layer in enumerate(self.backward_masks):
            if layer is None:  # No Sparisty, all active
                self.active_neurons.append([i for i in range(len(self.W[idx]))])
                continue
            active_neurons_this_layer = []
            for idx, filter_mask in enumerate(layer):
                if filter_mask.any():
                    active_neurons_this_layer.append(idx)
            self.active_neurons.append(active_neurons_this_layer)
        self.active_neuron_count = sum(
            [len(layer_neurons) for layer_neurons in self.active_neurons]
        )

    def _update_neuron_statistics(self) -> None:
        pass

    def state_dict(self):
        obj = {
            "dense_allocation": self.dense_allocation,
            "S": self.S,
            "N": self.N,
            "hyperparams": {
                "delta_T": self.delta_T,
                "alpha": self.alpha,
                "T_end": self.T_end,
                "ignore_linear_layers": self.ignore_linear_layers,
                "static_topo": self.static_topo,
                "sparsity_distribution": self.sparsity_distribution,
                "grad_accumulation_n": self.grad_accumulation_n,
                "erk_power_scale": self.erk_power_scale,
            },
            "step": self.step,
            "rigl_steps": self.rigl_steps,
            "backward_masks": self.backward_masks,
            "_linear_layers_mask": self._linear_layers_mask,
            "itop_rs": self.itop_rs,
            "explored_params": self.explored_params,
            "active_neurons": self.active_neurons,
        }

        return obj

    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            if type(v) == dict:
                self.load_state_dict(v)
            setattr(self, k, v)

    @torch.no_grad()
    def random_sparsify(self):
        is_dist = dist.is_initialized()
        self.backward_masks = []
        for l, w in enumerate(self.W):
            # if sparsity is 0%, skip
            if self.S[l] <= 0:
                self.backward_masks.append(None)
                continue

            n = self.N[l]
            s = int(self.S[l] * n)
            perm = torch.randperm(n)  # Generate random perm of indices in n
            perm = perm[
                :s
            ]  # Select s elements from n to achieve desired sparsity
            flat_mask = torch.ones(n, device=w.device)
            flat_mask[perm] = 0
            mask = torch.reshape(flat_mask, w.shape)

            if is_dist:
                dist.broadcast(mask, 0)

            mask = mask.bool()
            w *= mask
            self.backward_masks.append(mask)

    def __str__(self):
        s = "RigLScheduler(\n"
        s += "layers=%i,\n" % len(self.N)

        # calculate the number of non-zero elements out of the total number of
        # elements
        N_str = "["
        S_str = "["
        sparsity_percentages = []
        total_params = 0
        total_conv_params = 0
        total_nonzero = 0
        total_conv_nonzero = 0

        for N, S, mask, W, is_linear in zip(
            self.N,
            self.S,
            self.backward_masks,
            self.W,
            self._linear_layers_mask,
        ):
            actual_S = torch.sum(W[mask == 0] == 0).item()
            N_str += "%i/%i, " % (N - actual_S, N)
            sp_p = float(N - actual_S) / float(N) * 100
            S_str += "%.2f%%, " % sp_p
            sparsity_percentages.append(sp_p)
            total_params += N
            total_nonzero += N - actual_S
            if not is_linear:
                total_conv_nonzero += N - actual_S
                total_conv_params += N

        N_str = N_str[:-2] + "]"
        S_str = S_str[:-2] + "]"

        s += "nonzero_params=" + N_str + ",\n"
        s += "nonzero_percentages=" + S_str + ",\n"
        s += (
            "total_nonzero_params="
            + (
                "%i/%i (%.2f%%)"
                % (
                    total_nonzero,
                    total_params,
                    float(total_nonzero) / float(total_params) * 100,
                )
            )
            + ",\n"
        )
        s += (
            "total_CONV_nonzero_params="
            + (
                "%i/%i (%.2f%%)"
                % (
                    total_conv_nonzero,
                    total_conv_params,
                    float(total_conv_nonzero) / float(total_conv_params) * 100,
                )
            )
            + ",\n"
        )
        s += "step=" + str(self.step) + ",\n"
        s += "num_rigl_steps=" + str(self.rigl_steps) + ",\n"
        s += "ignoring_linear_layers=" + str(self.ignore_linear_layers) + ",\n"
        s += "sparsity_distribution=" + str(self.sparsity_distribution) + ",\n"
        s += "ITOP rate=" + f"{self.itop_rs:.4f}" + ",\n"
        active_neuron_count = [
            (len(self.active_neurons[idx]), self.W[idx].shape[0])
            for idx in range(len(self.W))
        ]
        s += "Active Neuron Count=" + f"{active_neuron_count}" + ",\n"
        return s + ")"

    @torch.no_grad()
    def reset_momentum(self):
        for w, mask, s in zip(self.W, self.backward_masks, self.S):
            # if sparsity is 0%, skip
            if s <= 0:
                continue

            param_state = self.optimizer.state[w]
            if "momentum_buffer" in param_state:
                # mask the momentum matrix
                buf = param_state["momentum_buffer"]
                buf *= mask

    @torch.no_grad()
    def apply_mask_to_weights(self):
        self._max_inactive_weights = []
        for w, mask, s in zip(self.W, self.backward_masks, self.S):
            # if sparsity is 0%, skip
            if s <= 0:
                self._max_inactive_weights.append(0.0)
                continue
            self._max_inactive_weights.append(
                torch.abs(w[mask == False]).max().item()  # noqa: E712
            )
            w *= mask

    @torch.no_grad()
    def apply_mask_to_gradients(self):
        for w, mask, s in zip(self.W, self.backward_masks, self.S):
            # if sparsity is 0%, skip
            if s <= 0:
                continue

            w.grad *= mask

    def check_if_backward_hook_should_accumulate_grad(self):
        """
        Used by the backward hooks. Basically just checks how far away the next
        rigl step is, if it's within `self.grad_accumulation_n` steps, return
        True.
        """

        if self.step >= self.T_end:
            return False

        steps_til_next_rigl_step = self.delta_T - (self.step % self.delta_T)
        return steps_til_next_rigl_step <= self.grad_accumulation_n

    def cosine_annealing(self):
        return self.alpha / 2 * (1 + np.cos((self.step * np.pi) / self.T_end))

    def __call__(self):
        self.step += 1
        if self.static_topo:
            return True
        if (
            self.step % self.delta_T
        ) == 0 and self.step < self.T_end:  # check schedule
            self._rigl_step()
            self.rigl_steps += 1
            self._update_itop_rs()
            # self._update_current_filter_ablation()
            self._update_active_neurons()
            return False
        return True

    @torch.no_grad()
    def _rigl_step(self):
        drop_fraction = self.cosine_annealing()

        # if distributed these values will be populated
        is_dist = dist.is_initialized()
        world_size = dist.get_world_size() if is_dist else None

        for idx, w in enumerate(self.W):
            # if sparsity is 0%, skip
            if self.S[idx] <= 0:
                continue

            current_mask = self.backward_masks[idx]

            # calculate raw scores
            score_drop = torch.abs(w)
            score_grow = torch.abs(self.backward_hook_objects[idx].dense_grad)

            # if is distributed, synchronize scores
            if is_dist:
                dist.all_reduce(score_drop)  # get the sum of all drop scores
                score_drop /= (
                    world_size  # divide by world size (average the drop scores)
                )

                dist.all_reduce(score_grow)  # get the sum of all grow scores
                score_grow /= (
                    world_size  # divide by world size (average the grow scores)
                )

            # calculate drop/grow quantities
            n_total = self.N[idx]
            n_ones = torch.sum(current_mask).item()
            n_prune = int(n_ones * drop_fraction)
            n_keep = n_ones - n_prune

            # create drop mask
            _, sorted_indices = torch.topk(score_drop.view(-1), k=n_total)
            new_values = torch.where(
                torch.arange(n_total, device=w.device) < n_keep,
                torch.ones_like(sorted_indices),
                torch.zeros_like(sorted_indices),
            )
            mask1 = new_values.scatter(0, sorted_indices, new_values)

            # flatten grow scores
            score_grow = score_grow.view(-1)

            # set scores of the enabled connections(ones) to min(s) - 1, so that
            # they have the lowest scores
            score_grow_lifted = torch.where(
                mask1 == 1,
                torch.ones_like(mask1) * (torch.min(score_grow) - 1),
                score_grow,
            )

            # create grow mask
            _, sorted_indices = torch.topk(score_grow_lifted, k=n_total)
            new_values = torch.where(
                torch.arange(n_total, device=w.device) < n_prune,
                torch.ones_like(sorted_indices),
                torch.zeros_like(sorted_indices),
            )
            mask2 = new_values.scatter(0, sorted_indices, new_values)

            mask2_reshaped = torch.reshape(mask2, current_mask.shape)
            grow_tensor = torch.zeros_like(w)

            REINIT_WHEN_SAME = False
            if REINIT_WHEN_SAME:
                raise NotImplementedError()
            else:
                new_connections = (mask2_reshaped == 1) & (current_mask == 0)

            # update new weights to be initialized as zeros and update the
            # weight tensors
            new_weights = torch.where(
                new_connections.to(w.device), grow_tensor, w
            )
            w.data = new_weights

            mask_combined = torch.reshape(
                mask1 + mask2, current_mask.shape
            ).bool()

            # update the mask
            current_mask.data = mask_combined

        self.reset_momentum()
        self.apply_mask_to_weights()
        self.apply_mask_to_gradients()

    def _update_itop_rs(self):
        if self.explored_params is None:
            self.explored_params = []
            for weight_tensor in self.W:
                self.explored_params.append(
                    torch.zeros(
                        size=weight_tensor.shape,
                        dtype=torch.bool,
                        device=weight_tensor.device,
                    )
                )
        for mask_idx in list(range(len(self.explored_params))):
            this_mask = self.backward_masks[mask_idx]
            if this_mask is None:
                this_mask = torch.ones(
                    size=self.W[mask_idx].shape,
                    dtype=torch.bool,
                    device=self.W[0].device,
                )
            self.explored_params[mask_idx] += this_mask
        for ep in self.explored_params:
            if ep is None:
                print("found empty ep")
        self.itop_rs = sum([ep.sum() for ep in self.explored_params]) / sum(
            [ep.numel() for ep in self.explored_params]
        )

    def _validate_percent_params(
        self, param_value: float, param_name: str
    ) -> bool:
        """Validate float parameters that are intended to be in range (0,1).

        Args:
            param_value (float): Value of parameter expected to be in range
                (0,1)
            param_name (str): Name of param, used for logging error to user.

        Raises:
            ValueError: If param value not in range 0,1

        Returns:
            bool: Returns True if param value is valid.
        """
        if param_value is None:
            return True
        if param_value <= 0 or param_value > 1:
            raise ValueError(
                f"{param_name} must be on the interval (0, 1]."
                f"Got: {param_value}"
            )
        return True

    def _update_current_filter_ablation(self) -> None:
        """Update list of ablated filters. TODO: Unused currently in favour of
            active neurons, consider removal

        Intended to monitor neuron ablation of vanilla rigl. Const-fan in rigl
        will have the same neuron ablations from initalization depending on
        value of cfg.rigl.filter_ablation_threshold
        """

        def get_num_ablated_filters(mask: Optional[torch.Tensor]) -> int:
            """Return number of filters in mask that are all False.

            Args:
                mask (Optional[torch.Tensor]): Mask from self.backward_masks.

            Returns:
                int: Number of filters with all elements == False.
            """
            if mask is None:
                return 0
            else:
                return torch.sum(
                    torch.stack([~filter.any() for filter in mask])
                )

        if not hasattr(self, "ablated_filters"):
            self.ablated_filters = []

        self.ablated_filters.append(
            [get_num_ablated_filters(filter) for filter in self.backward_masks]
        )
        return

    def get_global_sparsity_from_masks(self) -> float:
        """Return overall network sparsity based on backward mask values.

        Returns:
            float: Number of elements == False divided by total number of
                elements.
        """
        total_els = 0
        total_non_zero_els = 0
        for w, m in list(zip(self.W, self.backward_masks)):
            if m is None:
                total_non_zero_els += w.numel()
                total_els += w.numel()
            else:
                total_non_zero_els += m.sum()
                total_els += w.numel()
        return 1 - (total_non_zero_els / total_els)

    def _register_meters(self) -> None:
        self.meters = []
        names, weights = get_names_and_W(self.model)
        assert weights == self.W
        for idx, _ in enumerate(self.backward_masks):
            self.meters.append(
                LayerMeter(
                    idx=idx,
                    weight=weights[idx],
                    mask=self.backward_masks[idx],
                    name=names[idx],
                )
            )

    def log_meters(self, step: int) -> None:
        for idx, meter in enumerate(self.meters):
            if self.backward_hook_objects[idx] is None:
                dense_grad = self.W[idx].grad
            else:
                dense_grad = self.backward_hook_objects[idx].dense_grad
            meter.log_to_wandb(
                dense_grads=dense_grad,
                max_inactive_weights=self._max_inactive_weights,
                step=step,
            )
        total_neurons = sum([len(x) for x in self.W])
        wandb.log(
            {
                "_TOTAL_ACTIVE_NEURONS": self.active_neuron_count,
                "_TOTAL_PERCENTAGE_ACTIVE_NEURONS": self.active_neuron_count
                / total_neurons
                * 100,
                "_PRUNING_RATE": self.cosine_annealing(),
            },
            step=step,
        )


if __name__ == "__main__":
    from rigl_torch.datasets import get_dataloaders
    import hydra
    from rigl_torch.optim import CosineAnnealingWithLinearWarmUp
    from rigl_torch.models import ModelFactory
    from rigl_torch.rigl_constant_fan import RigLConstFanScheduler

    with hydra.initialize(config_path="../../configs"):
        cfg = hydra.compose(config_name="config.yaml", overrides=[])

    use_cuda = not cfg.compute.no_cuda and torch.cuda.is_available()
    torch.manual_seed(cfg.training.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    train_loader, test_loader = get_dataloaders(cfg)

    model = ModelFactory.load_model(
        model=cfg.model.name, dataset=cfg.dataset.name
    ).to(device)
    # model = get_model(cfg).to(device)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=cfg.training.lr)
    scheduler = CosineAnnealingWithLinearWarmUp(
        optimizer,
        T_max=cfg.training.epochs,
        eta_min=0,
        lr=cfg.training.lr,
        warm_up_steps=cfg.training.warm_up_steps,
    )
    pruner = lambda: True  # noqa: E731
    if cfg.rigl.dense_allocation is not None:
        T_end = int(0.75 * cfg.training.epochs * len(train_loader))
        if cfg.rigl.const_fan_in:
            rigl_scheduler = RigLConstFanScheduler
        else:
            rigl_scheduler = RigLScheduler
        pruner = rigl_scheduler(
            model,
            optimizer,
            dense_allocation=cfg.rigl.dense_allocation,
            alpha=cfg.rigl.alpha,
            delta=cfg.rigl.delta,
            static_topo=cfg.rigl.static_topo,
            T_end=T_end,
            ignore_linear_layers=False,
            grad_accumulation_n=cfg.rigl.grad_accumulation_n,
            sparsity_distribution=cfg.rigl.sparsity_distribution,
            erk_power_scale=cfg.rigl.erk_power_scale,
        )
    else:
        print(
            "cfg.rigl.dense_allocation is `null`, training with dense "
            "network..."
        )
    print(pruner)
