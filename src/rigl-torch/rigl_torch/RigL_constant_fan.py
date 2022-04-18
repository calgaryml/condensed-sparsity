""" implementation of https://arxiv.org/abs/1911.11134
    TODO: Docstrings, unit testing for new functions
"""

import numpy as np
import torch
import torch.distributed as dist

from rigl_torch.util import (
    get_W,
    calculate_fan_in_and_fan_out,
    get_fan_in_tensor,
)


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


class RigLConstFanScheduler:
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
    ):
        if dense_allocation <= 0 or dense_allocation > 1:
            raise Exception(
                "Dense allocation must be on the interval (0, 1]. Got: %f"
                % dense_allocation
            )

        self.model = model
        self.optimizer = optimizer

        self.W, self._linear_layers_mask = get_W(
            model, return_linear_layers_mask=True
        )

        # modify optimizer.step() function to call "reset_momentum" after
        _create_step_wrapper(self, optimizer)

        self.dense_allocation = dense_allocation
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
            self.S = []
            for i, (W, is_linear) in enumerate(
                zip(self.W, self._linear_layers_mask)
            ):
                # when using uniform sparsity, the first layer is always 100%
                # dense UNLESS there is only 1 layer
                is_first_layer = i == 0
                if (
                    is_first_layer
                    and self.sparsity_distribution == "uniform"
                    and len(self.W) > 1
                ):
                    self.S.append(0)

                elif is_linear and self.ignore_linear_layers:
                    # if choosing to ignore linear layers, keep them 100% dense
                    self.S.append(0)

                else:
                    self.S.append(1 - dense_allocation)

            # randomly sparsify model according to S
            self.random_sparsify()

            # scheduler keeps a log of how many times it's called. this is how
            # it does its scheduling
            self.step = 0
            self.rigl_steps = 0

            # define the actual schedule
            self.delta_T = delta
            self.alpha = alpha
            self.T_end = T_end

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

        assert self.grad_accumulation_n > 0 and self.grad_accumulation_n < delta
        assert self.sparsity_distribution in (
            "uniform",
        )  # TODO: Implement ERK distribution

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
            },
            "step": self.step,
            "rigl_steps": self.rigl_steps,
            "backward_masks": self.backward_masks,
            "_linear_layers_mask": self._linear_layers_mask,
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

            fan_in, _ = calculate_fan_in_and_fan_out(w)
            s = int(fan_in * self.S[l])  # Number of connections to drop
            perm = torch.concat(
                [
                    torch.randperm(fan_in).reshape(1, -1)
                    for i in range(w.shape[0])
                ]
            )
            # Generate random perm of indices to mask per filter / neuron
            perm = perm[
                :, :s
            ]  # Drop s elements from n to achieve desired sparsity
            mask = torch.concat(
                [torch.ones(fan_in).reshape(1, -1) for i in range(w.shape[0])]
            )
            for m in range(mask.shape[0]):  # TODO: vectorize
                mask[m][perm[m]] = 0
            mask = mask.reshape(w.shape).to(device=w.device)

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
        const_fan_ins = []

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
            if mask is None:
                fan_in, _ = calculate_fan_in_and_fan_out(W)
                const_fan_ins.append(fan_in)
            else:
                const_fan_ins.append(get_fan_in_tensor(mask).unique().item())

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
        s += "constant fan ins=" + str(const_fan_ins) + ",\n"

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
        for w, mask, s in zip(self.W, self.backward_masks, self.S):
            # if sparsity is 0%, skip
            if s <= 0:
                continue

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
            return False
        return True

    @torch.no_grad()
    def _rigl_step(self):
        # TODO: Make object for mask?
        drop_fraction = self.cosine_annealing()
        # if distributed these values will be populated
        is_dist = dist.is_initialized()
        world_size = dist.get_world_size() if is_dist else None

        for l, w in enumerate(self.W):
            # if sparsity is 0%, skip
            if self.S[l] <= 0:
                continue

            # calculate raw scores
            score_drop = torch.abs(w)
            score_grow = torch.abs(self.backward_hook_objects[l].dense_grad)

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

            current_mask = self.backward_masks[l]

            # calculate drop/grow quantities
            n_fan_in = get_fan_in_tensor(current_mask).unique().item()
            n_ones = torch.sum(current_mask).item()
            n_prune = int(n_ones * drop_fraction)
            n_keep = n_ones - n_prune
            n_non_zero_weights = (score_drop != 0).sum().item()
            if n_non_zero_weights < n_keep:
                # Then we don't have enough non-zero weights to keep. We keep
                # ALL non-zero weights in this scenario and readjust our keep /
                # prune amounts to suit
                n_keep = n_non_zero_weights
                n_prune = n_ones - n_keep

            # create drop mask
            drop_mask = self._get_drop_mask(score_drop, n_keep)

            # create growth mask per filter
            grow_mask = self._get_grow_mask(score_grow, drop_mask, n_fan_in)

            # get new weights
            new_weights = self._get_new_weights(w, current_mask, grow_mask)
            w.data = new_weights

            combined_mask = grow_mask + drop_mask
            current_mask.data = combined_mask

            self.reset_momentum()
            self.apply_mask_to_weights()
            self.apply_mask_to_gradients()

    def _get_drop_mask(
        self, score_drop: torch.Tensor, n_keep: int
    ) -> torch.Tensor:
        idx_to_not_drop = torch.topk(score_drop.flatten(), k=n_keep).indices
        drop_mask = torch.zeros(size=(score_drop.numel(),), dtype=torch.bool)
        drop_mask[idx_to_not_drop] = True
        drop_mask = drop_mask.reshape(score_drop.shape)
        return drop_mask.to(device=score_drop.device)

    def _get_grow_mask(
        self,
        score_grow,
        drop_mask,
        n_fan_in,
    ):

        grow_mask = torch.zeros(
            size=drop_mask.shape, dtype=torch.bool, device=drop_mask.device
        )
        for idx, (drop_mask_filter, grow_mask_filter) in enumerate(
            list(zip(drop_mask, grow_mask))
        ):  # Iterate over filters
            if drop_mask_filter.sum() < n_fan_in:
                # set scores of the enabled connections(ones) to min(s) - 1,
                # so that they have the lowest scores
                score_grow_lifted = torch.where(
                    drop_mask_filter == True,  # noqa: E712
                    torch.ones_like(drop_mask_filter)
                    * (torch.min(score_grow[idx]) - 1),
                    score_grow[idx],
                )
                # Set currently active connections to min score to avoid
                # reselecting them
                idx_to_grow = torch.topk(
                    score_grow_lifted.flatten(),
                    k=n_fan_in - drop_mask_filter.sum(),
                ).indices
                # Grow enough connections to get to n_fan_in
                grow_mask_filter = grow_mask_filter.flatten()
                grow_mask_filter[idx_to_grow] = True
                grow_mask[idx] = grow_mask_filter.reshape(drop_mask[idx].shape)
            elif drop_mask_filter.sum() > n_fan_in:
                print(get_fan_in_tensor(drop_mask))
                raise ValueError(
                    f"Filter with {drop_mask_filter.sum()} fan in > than ",
                    "n_fan_in ({n_fan_in})",
                )
        assert (get_fan_in_tensor(drop_mask + grow_mask) == n_fan_in).all()
        return grow_mask

    def _get_new_weights(self, w, current_mask, grow_mask):
        grow_tensor = torch.zeros_like(w)
        new_connections = (current_mask == 0) & (
            grow_mask.to(device=current_mask.device) == 1
        )
        new_weights = torch.where(
            new_connections,
            grow_tensor,  # init to 0
            w,  # else keep existing weight
        )
        return new_weights
