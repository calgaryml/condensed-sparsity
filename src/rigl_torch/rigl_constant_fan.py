""" implementation of https://arxiv.org/abs/1911.11134
"""
from typing import Optional, Dict, Any, List
import torch
import torch.distributed as dist
from rigl_torch.utils.rigl_utils import (
    get_fan_in_tensor,
    get_fan_in_after_ablation,
    calculate_fan_in_and_fan_out,
)
from rigl_torch.rigl_scheduler import RigLScheduler
from rigl_torch.exceptions import (
    ConstantFanInException,
    InvalidAblatedNeuronException,
)


class RigLConstFanScheduler(RigLScheduler):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        dense_allocation: int = 1,
        T_end: Optional[int] = None,
        sparsity_distribution: str = "uniform",
        ignore_linear_layers: bool = False,
        delta: int = 100,
        alpha: float = 0.3,
        static_topo: bool = False,
        grad_accumulation_n: int = 1,
        state_dict: Optional[Dict[str, Any]] = None,
        erk_power_scale=1.0,
        filter_ablation_threshold: Optional[float] = None,
        static_ablation: bool = False,
        dynamic_ablation: bool = False,
        min_salient_weights_per_neuron: int = 0,
    ):
        """RigL Scheduler with constant fan-in.

        Constant fan-in enforced at initalization and during each grow / prune
        step.

        Args:
            model (torch.nn.Module): Model to sparsify.
            optimizer (torch.optim.Optimizer): Optimizer to wrap with rigl
                scheduler
            dense_allocation (int, optional): Percentage of dense parameters
                allowed. if None, pruning will not be used. must be on the
                interval (0, 1]". Defaults to 1.
            T_end (Optional[int], optional): Number of epochs to simulate (only
                used for tuning). Defaults to None.
            sparsity_distribution (str, optional): Description of sparsity
                distribution. Defaults to "uniform".
            ignore_linear_layers (bool, optional): If True, linear layers are
                not sparsified. Defaults to False.
            delta (int, optional): Delta param for pruning. Defaults to 100.
            alpha (float, optional): Alpha param for pruning. Defaults to 0.3.
            static_topo (bool, optional): If True, use random sparsity topo and
                remain static. Defaults to False.
            grad_accumulation_n (int, optional): Number of gradients to
                accumulate before scoring for rigl. Defaults to 1.
            state_dict (Dict[str, Any], optional): State dict used to initalize
                from rigl scheduler already initalized / trained. Defaults to
                None.
        """
        super().__init__(
            model,
            optimizer,
            dense_allocation,
            T_end,
            sparsity_distribution,
            ignore_linear_layers,
            delta,
            alpha,
            static_topo,
            grad_accumulation_n,
            state_dict,
            erk_power_scale,
            filter_ablation_threshold,
            static_ablation,
            dynamic_ablation,
            min_salient_weights_per_neuron,
        )
        self._dynamically_ablated_neuron_idx = [[] for _ in range(len(self.W))]

    @torch.no_grad()
    def random_sparsify(self) -> None:
        """Randomly sparsifies model to desired sparsity distribution with
        constant fan in.
        """
        is_dist: bool = dist.is_initialized()
        self.backward_masks: List[torch.tensor] = []
        for idx, (w, num_neurons_to_ablate) in enumerate(
            list(zip(self.W, self.static_ablated_filters))
        ):
            # if sparsity is 0%, skip
            if self.S[idx] <= 0:
                self.backward_masks.append(None)
                continue

            dense_fan_in, _ = calculate_fan_in_and_fan_out(module=w)
            fan_in = get_fan_in_after_ablation(
                weight_tensor=w,
                num_neurons_to_ablate=num_neurons_to_ablate,
                sparsity=self.S[idx],
            )
            # Number of connections to drop per filter
            s = dense_fan_in - fan_in
            perm = torch.concat(
                [
                    torch.randperm(dense_fan_in).reshape(1, -1)
                    for _ in range(w.shape[0])
                ]
            )
            # Generate random perm of indices to mask per filter / neuron
            perm = perm[
                :, :s
            ]  # Drop s elements from n to achieve desired sparsity
            mask = torch.concat(
                [
                    torch.ones(dense_fan_in).reshape(1, -1)
                    for _ in range(w.shape[0])
                ]
            )
            for filter_idx in range(mask.shape[0]):  # TODO: vectorize?
                mask[filter_idx][perm[filter_idx]] = 0
            mask = mask.reshape(w.shape).to(device=w.device)
            # Ablate top n neurons according to filter sparsity criterion
            mask[:num_neurons_to_ablate] = 0

            if is_dist:
                dist.broadcast(mask, 0)
            mask = mask.bool()
            w *= mask
            self.backward_masks.append(mask)

    def __str__(self) -> str:
        """Appends constant fan in info to RigL scheduler __str__.

        Raises:
            ConstantFanInException: If constant fan-in inconsistent for any
                mask.

        Returns:
            str: String describing state of scheduler.
        """
        s = super().__str__()
        s = s[:-1]  # Remove trailing ')'
        const_fan_ins = []
        for mask, w, neurons_ablated in zip(
            self.backward_masks,
            self.W,
            self._dynamically_ablated_neuron_idx,
        ):
            if mask is None:
                fan_in, _ = calculate_fan_in_and_fan_out(w)
                const_fan_ins.append(fan_in)
            else:
                try:
                    active_filters = [
                        i for i in range(len(w)) if i not in neurons_ablated
                    ]
                    const_fan_ins.append(
                        get_fan_in_tensor(mask[active_filters]).unique().item()
                    )

                except ValueError:
                    raise ConstantFanInException(
                        get_fan_in_tensor(mask[active_filters])
                    )

        s = f"{s}constant fan ins={str(const_fan_ins)}\n"
        s = (
            f"{s}Neurons Statically Ablated per layer = "
            f"{str(self.static_ablated_filters)}\n"
        )
        s = (
            f"{s}Neurons Dynamically Ablated per layer = "
            f"{str([len(x) for x in self._dynamically_ablated_neuron_idx])}\n)"
        )
        return s

    @torch.no_grad()
    def _rigl_step(self) -> None:
        """Performs rigl update prune / regrowth with constant fan-in.

        Raises:
            ConstantFanInException: If constant fan-in inconsistent for any
                mask.
        """
        drop_fraction = self.cosine_annealing()
        # if distributed these values will be populated
        is_dist = dist.is_initialized()
        world_size = dist.get_world_size() if is_dist else None

        self._dynamically_ablated_neuron_idx = []
        for idx, w in enumerate(self.W):
            # if sparsity is 0%, skip
            if self.S[idx] <= 0:
                continue

            # calculate raw scores
            score_drop = torch.abs(w)
            _max_score_drop = score_drop.max().item()

            # Set ablated filter drop scores to min of score_grow to avoid
            # pruning already inactive weights
            # TODO: Remove inital ablated filtering.
            score_drop[
                : self.static_ablated_filters[idx]
            ] = score_drop.min().item()

            score_grow = torch.abs(self.backward_hook_objects[idx].dense_grad)

            # Set ablated filter scores to min of score_grow to avoid regrowing
            score_grow[
                : self.static_ablated_filters[idx]
            ] = score_grow.min().item()

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

            current_mask = self.backward_masks[idx]
            n_total = self.N[idx]
            n_ones = torch.sum(current_mask).item()
            n_prune = int(n_ones * drop_fraction)
            n_keep = int(n_ones - n_prune)
            n_non_zero_weights = torch.count_nonzero(score_drop).item()
            if n_non_zero_weights < n_keep:
                # Then we don't have enough non-zero weights to keep. We keep
                # ALL non-zero weights in this scenario and readjust our keep /
                # prune amounts to suit
                n_keep = n_non_zero_weights
                n_prune = n_ones - n_keep

            # Get neurons to ablate
            neurons_to_ablate = self._get_neurons_to_ablate(
                score_drop=score_drop,
                score_grow=score_grow,
                n_keep=n_keep,
                n_prune=n_prune,
                n_total=n_total,
            )
            self._dynamically_ablated_neuron_idx.append(neurons_to_ablate)
            # print(f"neurons to ablate = {neurons_to_ablate}")
            # print(f"len neurons to ablate = {len(neurons_to_ablate)}")
            n_fan_in = get_fan_in_after_ablation(
                weight_tensor=w,
                num_neurons_to_ablate=len(neurons_to_ablate),
                sparsity=self.S[idx],
            )

            # create drop mask
            drop_mask = self._get_drop_mask(
                score_drop,
                n_keep,
                neurons_to_ablate=neurons_to_ablate,
                n_fan_in=n_fan_in,
            )

            # create growth mask per filter
            grow_mask = self._get_grow_mask(
                score_grow,
                drop_mask,
                n_fan_in,
                neurons_to_ablate,
            )

            # get new weights
            new_weights = self._get_new_weights(w, current_mask, grow_mask)
            w.data = new_weights

            combined_mask = grow_mask + drop_mask
            current_mask.data = combined_mask

            self.reset_momentum()
            self.apply_mask_to_weights()
            self.apply_mask_to_gradients()
            self._verify_neuron_ablation()
            if (
                self.min_salient_weights_per_neuron == 1
                and torch.abs(w).max().item() != _max_score_drop
            ):
                self._logger.warning(
                    "Max score drop not equal to max weight after masking! "
                    f"I have a pre-mask max of {_max_score_drop} and a post "
                    f"mask max of {torch.abs(w).max().item()}"
                )
            # print("rigl step passed")

    @torch.no_grad()
    def _get_neurons_to_ablate(
        self,
        score_drop: torch.Tensor,
        score_grow,
        n_keep,
        n_prune,
        n_total,
    ) -> List[int]:
        if self.dynamic_ablation:
            neurons_to_ablate: List[int] = []
            saliency_mask = torch.zeros(
                size=(score_drop.numel(),),
                dtype=torch.bool,
                device=score_drop.device,
            )
            _, keep_idx = score_drop.flatten().sort(descending=True)
            saliency_mask[keep_idx[:n_keep]] = True

            _, grow_idx = score_grow.flatten().sort(descending=True)
            saliency_mask[grow_idx[:n_prune]] = True

            saliency_mask = saliency_mask.reshape(shape=score_drop.shape)
            for neuron_idx, neuron in enumerate(saliency_mask):
                if neuron.sum() < self.min_salient_weights_per_neuron:
                    neurons_to_ablate.append(neuron_idx)
            return neurons_to_ablate

        if self.static_ablation:
            return (
                self.static_ablated_filters
            )  # Check type -> Need to convert to list of indices

    @torch.no_grad()
    def _get_drop_mask(
        self,
        score_drop: torch.Tensor,
        n_keep: int,
        neurons_to_ablate: List[int],
        n_fan_in: int,
    ) -> torch.Tensor:
        """Gets weights to prune by selecting -abs(score_drop).

        Args:
            score_drop (torch.Tensor): Weight magnitude tensor
            n_keep (int): Number of connections to keep.

        Raises:
            RuntimeError: If n_keep > score_drop.numel().

        Returns:
            torch.Tensor: Boolean mask of connections to keep. Where True, keep
                connection else prune.
        """
        # Set ablated neuron scores to min
        for neuron_idx in range(len(score_drop)):
            if neuron_idx in neurons_to_ablate:
                score_drop[neuron_idx] = score_drop.min() - 1
        try:
            idx_to_not_drop = torch.topk(
                score_drop.flatten(), k=n_keep, sorted=False
            ).indices
        except RuntimeError as e:
            self._logger.error(
                f"n_keep > score_drop.numel() ({n_keep} > {score_drop.numel()})"
            )
            raise RuntimeError(
                "RigLConstFanScheduler._get_drop_mask: n_keep > "
                "score_drop.numel()"
            ) from e

        drop_mask = torch.zeros(size=(score_drop.numel(),), dtype=torch.bool)
        drop_mask[idx_to_not_drop] = True
        drop_mask = drop_mask.reshape(score_drop.shape)

        # In some cases, one neuron may have more than n_fan_in salient weights
        # So we sort each filter and ensure that weights not in topk are maksed
        # to False
        # START HERE
        for idx, neuron_mask in enumerate(drop_mask):
            _, sorted_idx = neuron_mask.flatten().sort(descending=True)
            neuron_mask_shape = neuron_mask.shape
            neuron_mask = neuron_mask.flatten()
            neuron_mask[sorted_idx[n_fan_in:]] = False
            drop_mask[idx] = neuron_mask.reshape(neuron_mask_shape)

        return drop_mask.to(device=score_drop.device)

    @torch.no_grad()
    def _get_grow_mask(
        self,
        score_grow: torch.Tensor,
        drop_mask: torch.Tensor,
        n_fan_in: int,
        neurons_to_ablate: List[int],
    ) -> torch.Tensor:
        """Get weights to grow by selecting abs(score_grow) where not already
            active with constant fan-in.

        Args:
            score_grow (torch.Tensor): Absolute value of dense gradients.
            drop_mask (torch.Tensor): Boolean mask from _get_drop_mask(). Where
                True, connections are active.
            n_fan_in (int): Number of connections to grow.

        Raises:
            ValueError: If constant fan-in requirement is voided by union of
                drop_mask and growth_mask.

        Returns:
            torch.Tensor: Boolean tensor of weights to grow.
        """

        grow_mask = torch.zeros(
            size=drop_mask.shape, dtype=torch.bool, device=drop_mask.device
        )
        for idx, (drop_mask_filter, grow_mask_filter) in enumerate(
            list(zip(drop_mask, grow_mask))
        ):  # Iterate over filters
            if idx in neurons_to_ablate:
                grow_mask_filter[:] = False
                grow_mask[idx] = grow_mask_filter
            elif drop_mask_filter.sum() < n_fan_in:
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
            elif (
                drop_mask_filter.sum() > n_fan_in
            ):  # TODO: Should handle this case in drop mask
                self._logger.error(
                    get_fan_in_tensor(
                        drop_mask[self.static_ablated_filters[idx] :]  # noqa
                    )
                )
                raise ValueError(
                    f"Filter with {drop_mask_filter.sum()} fan in > than ",
                    "n_fan_in ({n_fan_in})",
                )
        # TODO need inverse select
        should_be_active_neurons = [
            i for i in range(len(grow_mask)) if i not in neurons_to_ablate
        ]
        assert (
            get_fan_in_tensor(
                drop_mask[should_be_active_neurons]
                + grow_mask[should_be_active_neurons]
            )
            == n_fan_in
        ).all()
        return grow_mask

    @property
    def inverse_idx_select(self):
        active_neurons = []
        for layer_idx, layer in enumerate(self._dynamically_ablated_neuron_idx):
            active_neurons.append(
                [i for i in range(len(self.W[layer_idx])) if i not in layer]
            )
        return active_neurons

    def _get_new_weights(
        self,
        w: torch.nn.parameter.Parameter,
        current_mask: torch.Tensor,
        grow_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Gets new weights for grown connections.

        New weights initalized to 0, otherwise previous weight value retained.

        Args:
            w (torch.nn.parameter.Parameter): Weight matrix for a given layer
            current_mask (torch.Tensor): Current mask from last step for a given
                layer.
            grow_mask (torch.Tensor): New grow_mask obtained in this rigl step.
                Where True, weights initalized to zero.

        Returns:
            torch.Tensor: New weight matrix with zeros for newly grown weights.
        """
        grow_tensor = torch.zeros_like(w)
        new_connections = ~current_mask & grow_mask.to(
            device=current_mask.device
        )
        new_weights = torch.where(
            new_connections,
            grow_tensor,  # init to 0
            w,  # else keep existing weight
        )
        return new_weights

    def _verify_neuron_ablation(self) -> None:
        """Verify that backward_masks do not have any active elements i
        initally ablated filters.

        Raises:
            InvalidAblatedNeuronException: If an ablated filter has a value !=
                False.
        """
        for mask_index, (m, n) in enumerate(
            list(zip(self.backward_masks, self.static_ablated_filters))
        ):
            if m is None:
                continue
            else:
                if not ~(m[:n].any()):
                    raise InvalidAblatedNeuronException(mask_index)
