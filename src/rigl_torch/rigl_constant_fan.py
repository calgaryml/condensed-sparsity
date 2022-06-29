""" implementation of https://arxiv.org/abs/1911.11134
"""
from typing import Optional, Dict, Any, List
import torch
import torch.distributed as dist
from rigl_torch.util import (
    calculate_fan_in_and_fan_out,
    get_fan_in_tensor,
)
from rigl_torch.rigl_scheduler import RigLScheduler
from rigl_torch.exceptions import ConstantFanInException


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
        )

    @torch.no_grad()
    def random_sparsify(self) -> None:
        """Randomly sparsifies model to desired sparsity distribution with
        constant fan in.
        """
        is_dist: bool = dist.is_initialized()
        self.backward_masks: List[torch.tensor] = []
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
            for m in range(mask.shape[0]):  # TODO: vectorize?
                mask[m][perm[m]] = 0
            mask = mask.reshape(w.shape).to(device=w.device)

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
        for mask, W, in zip(self.backward_masks, self.W,):
            if mask is None:
                fan_in, _ = calculate_fan_in_and_fan_out(W)
                const_fan_ins.append(fan_in)
            else:
                try:
                    const_fan_ins.append(
                        get_fan_in_tensor(mask).unique().item()
                    )
                except ValueError:
                    raise ConstantFanInException(get_fan_in_tensor(mask))

        s = f"{s}constant fan ins={str(const_fan_ins)}\n)"
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
            try:
                n_fan_in = get_fan_in_tensor(current_mask).unique().item()
            except ValueError:
                raise ConstantFanInException(get_fan_in_tensor(current_mask))
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
        return drop_mask.to(device=score_drop.device)

    def _get_grow_mask(
        self, score_grow: torch.Tensor, drop_mask: torch.Tensor, n_fan_in: int,
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
                self._logger.error(get_fan_in_tensor(drop_mask))
                raise ValueError(
                    f"Filter with {drop_mask_filter.sum()} fan in > than ",
                    "n_fan_in ({n_fan_in})",
                )
        assert (get_fan_in_tensor(drop_mask + grow_mask) == n_fan_in).all()
        return grow_mask

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
        # new_connections = (current_mask == 0) & (
        #     grow_mask.to(device=current_mask.device) == 1
        # )
        new_weights = torch.where(
            new_connections,
            grow_tensor,  # init to 0
            w,  # else keep existing weight
        )
        return new_weights
