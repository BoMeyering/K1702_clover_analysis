"""
src.ssl.dataloaders.py
Dataloader and data sampler functions
Ported from pgcview_v2_training_framework/src/dataloaders.py.
Changes: DistributedInfiniteSampler reshuffles on every pass; num_workers is a parameter.
BoMeyering 2025
"""

import math

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler


class DistributedInfiniteSampler(Sampler[int]):
    def __init__(
        self,
        dataset: Dataset,
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ):
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        # Only require dist init if we need to infer rank/world_size
        if num_replicas is None or rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            if not dist.is_initialized():
                raise RuntimeError("Distributed package is not initialized")
            if num_replicas is None:
                num_replicas = dist.get_world_size()
            if rank is None:
                rank = dist.get_rank()

        if num_replicas is None or rank is None:
            raise ValueError(
                "num_replicas and rank must be set (either passed in or via torch.distributed)"
            )

        if rank < 0 or rank >= num_replicas:
            raise ValueError(
                f"Invalid rank {rank}, expected 0 <= rank < {num_replicas}"
            )

        self.num_replicas = num_replicas
        self.rank = rank

        n = len(self.dataset)
        if self.drop_last and n < self.num_replicas:
            raise ValueError(
                f"drop_last=True but dataset length ({n}) < num_replicas ({self.num_replicas}); "
                "this would yield zero samples per rank."
            )

        if self.drop_last:
            self.num_samples = n // self.num_replicas
        else:
            self.num_samples = int(math.ceil(n / self.num_replicas))

        if self.num_samples == 0:
            raise ValueError(
                "DistributedInfiniteSampler computed num_samples=0; check dataset length/drop_last."
            )

        self.total_size = self.num_samples * self.num_replicas

    def _rank_indices(self, pass_idx: int) -> list[int]:
        n = len(self.dataset)

        if self.shuffle:
            # Same seed on every rank so the shards stay disjoint; new permutation on every pass
            g = torch.Generator()
            g.manual_seed(self.seed + 1_000_003 * self.epoch + pass_idx)
            indices = torch.randperm(n, generator=g).tolist()
        else:
            indices = list(range(n))

        if not self.drop_last:
            if len(indices) < self.total_size:
                indices += indices[: (self.total_size - len(indices))]
            else:
                indices = indices[: self.total_size]
        else:
            indices = indices[: self.total_size]

        indices = indices[self.rank : self.total_size : self.num_replicas]
        if len(indices) != self.num_samples:
            raise RuntimeError(
                f"Expected {self.num_samples} indices for rank {self.rank}, got {len(indices)}"
            )
        return indices

    def __iter__(self):
        pass_idx = 0
        while True:
            yield from self._rank_indices(pass_idx)
            pass_idx += 1

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
