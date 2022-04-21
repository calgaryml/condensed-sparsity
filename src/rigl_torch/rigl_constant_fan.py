""" implementation of https://arxiv.org/abs/1911.11134
"""

from typing import Optional, Dict, Any
import torch
import torch.distributed as dist
from rigl_torch.util import (
    calculate_fan_in_and_fan_out,
    get_fan_in_tensor,
)
from rigl_torch.RigL import RigLScheduler


class RigLConstFanScheduler(RigLScheduler):
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        dense_allocation: int = 1,
        T_end: Optional[int] = None,
        sparsity_distribution: str = "uniform",
        ignore_linear_layers: bool = True,
        delta: int = 100,
        alpha: float = 0.3,
        static_topo: bool = False,
        grad_accumulation_n: int = 1,
        state_dict: Dict[str, Any] = None,
    ):
        """RigL Scheduler with constant fan-in.

        Constant fan-in enforced at initalization and during each grow / prune
        step.

        Args:
            model (torch.nn.Module): _description_
            optimizer (torch.optim.Optimizer): _description_
            dense_allocation (int, optional): _description_. Defaults to 1.
            T_end (Optional[int], optional): _description_. Defaults to None.
            sparsity_distribution (str, optional): _description_. Defaults to
                "uniform".
            ignore_linear_layers (bool, optional): _description_. Defaults to
                True.
            delta (int, optional): _description_. Defaults to 100.
            alpha (float, optional): _description_. Defaults to 0.3.
            static_topo (bool, optional): _description_. Defaults to False.
            grad_accumulation_n (int, optional): _description_. Defaults to 1.
            state_dict (Dict[str, Any], optional): _description_. Defaults to
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
        )
        # TODO: Implement ERK distribution

    @torch.no_grad()
    def random_sparsify(self):
        """Randomly sparsify model to desired sparsity distribution with
        constant fan in.
        """
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
            for m in range(mask.shape[0]):  # TODO: vectorize?
                mask[m][perm[m]] = 0
            mask = mask.reshape(w.shape).to(device=w.device)

            if is_dist:
                dist.broadcast(mask, 0)

            mask = mask.bool()
            w *= mask
            self.backward_masks.append(mask)

    def __str__(self):
        s = super().__str__()
        s = s[:-1]  # Remove trailing ')'
        const_fan_ins = []
        for mask, W, in zip(
            self.backward_masks,
            self.W,
        ):
            if mask is None:
                fan_in, _ = calculate_fan_in_and_fan_out(W)
                const_fan_ins.append(fan_in)
            else:
                const_fan_ins.append(get_fan_in_tensor(mask).unique().item())
        s += "constant fan ins=" + str(const_fan_ins) + ",\n"
        return s + ")"

    @torch.no_grad()
    def _rigl_step(self):
        """Perform rigl update prune / regrowth with constant fan-in."""
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
        """Get weights to prune by selecting -abs(score_drop) (weight magnitude)

        Args:
            score_drop (torch.Tensor): Weight magnitude tensor
            n_keep (int): Number of connections to keep.

        Returns:
            torch.Tensor: Boolean mask of connections to keep. Where True, keep
                connection else prune.
        """

        idx_to_not_drop = torch.topk(score_drop.flatten(), k=n_keep).indices
        drop_mask = torch.zeros(size=(score_drop.numel(),), dtype=torch.bool)
        drop_mask[idx_to_not_drop] = True
        drop_mask = drop_mask.reshape(score_drop.shape)
        return drop_mask.to(device=score_drop.device)

    def _get_grow_mask(
        self,
        score_grow: torch.Tensor,
        drop_mask: torch.Tensor,
        n_fan_in: int,
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
