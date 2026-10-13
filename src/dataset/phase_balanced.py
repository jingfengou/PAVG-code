"""Deterministic interleaving of solid and gas Stage 1 samples."""

from __future__ import annotations

from typing import Dict, Iterator, List, Sequence
import hashlib
import json

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


class PhaseBalancedDataset(Dataset):
    """Expose two trajectory datasets through one phase-tagged index space."""

    SOLID = 0
    GAS = 1

    def __init__(self, solid_dataset: Dataset, gas_dataset: Dataset):
        if len(solid_dataset) == 0 or len(gas_dataset) == 0:
            raise ValueError("Both phase datasets must be non-empty")
        self.solid_dataset = solid_dataset
        self.gas_dataset = gas_dataset
        self.solid_size = len(solid_dataset)
        self.gas_size = len(gas_dataset)

    def __len__(self) -> int:
        return self.solid_size + self.gas_size

    def __getitem__(self, index: int):
        index = int(index)
        if not 0 <= index < len(self):
            raise IndexError(index)
        if index < self.solid_size:
            sample, info = self.solid_dataset[index]
            phase_id = self.SOLID
            phase_index = index
            phase_name = "solid"
        else:
            phase_index = index - self.solid_size
            sample, info = self.gas_dataset[phase_index]
            phase_id = self.GAS
            phase_name = "gas"

        sample = dict(sample)
        info = dict(info)
        sample["phase_id"] = torch.tensor(phase_id, dtype=torch.long)
        info["phase"] = phase_name
        info["phase_index"] = phase_index
        return sample, info


class PhaseBalancedBatchSampler(Sampler[List[int]]):
    """Yield all local batches before Accelerate assigns whole batches to ranks."""

    def __init__(
        self,
        *,
        solid_count: int,
        gas_count: int,
        target_solid_examples: int,
        target_gas_examples: int,
        solid_per_batch: int,
        gas_per_batch: int,
        sampler_seed: int,
        world_size: int,
        optimizer_steps: int,
        gradient_accumulation_steps: int,
    ):
        self.solid_count = int(solid_count)
        self.gas_count = int(gas_count)
        self.target_solid_examples = int(target_solid_examples)
        self.target_gas_examples = int(target_gas_examples)
        self.solid_per_batch = int(solid_per_batch)
        self.gas_per_batch = int(gas_per_batch)
        self.sampler_seed = int(sampler_seed)
        self.world_size = int(world_size)
        self.optimizer_steps = int(optimizer_steps)
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)

        if self.solid_count <= 0 or self.gas_count <= 0:
            raise ValueError("Phase manifest counts must be positive")
        if self.solid_per_batch <= 0 or self.gas_per_batch <= 0:
            raise ValueError("Per-batch phase counts must be positive")
        if self.world_size <= 0 or self.optimizer_steps <= 0:
            raise ValueError("world_size and optimizer_steps must be positive")
        if self.gradient_accumulation_steps != 1:
            raise ValueError(
                "Exposure-matched protocol requires gradient_accumulation_steps=1"
            )
        if self.target_solid_examples % self.solid_per_batch:
            raise ValueError("Solid exposure is not divisible by solid_per_batch")
        if self.target_gas_examples % self.gas_per_batch:
            raise ValueError("Gas exposure is not divisible by gas_per_batch")

        self.solid_batches = self.target_solid_examples // self.solid_per_batch
        self.gas_batches = self.target_gas_examples // self.gas_per_batch
        if self.solid_batches != self.gas_batches:
            raise ValueError(
                "Solid and gas exposures do not produce the same number of local batches"
            )
        self.global_batch_count = self.solid_batches
        expected_global_batches = self.optimizer_steps * self.world_size
        if self.global_batch_count != expected_global_batches:
            raise ValueError(
                "Exposure target does not match optimizer_steps * world_size: "
                f"batches={self.global_batch_count} expected={expected_global_batches}"
            )
        if self.global_batch_count % self.world_size:
            raise ValueError("Global batch count must shard evenly across ranks")

    @property
    def local_batch_size(self) -> int:
        return self.solid_per_batch + self.gas_per_batch

    @property
    def local_batches_per_rank(self) -> int:
        return self.global_batch_count // self.world_size

    def __len__(self) -> int:
        return self.global_batch_count

    def _index_stream(self, count: int, target: int, seed: int) -> Iterator[int]:
        rng = np.random.default_rng(seed)
        emitted = 0
        while emitted < target:
            permutation = rng.permutation(count)
            take = min(count, target - emitted)
            for value in permutation[:take]:
                yield int(value)
            emitted += take

    def __iter__(self) -> Iterator[List[int]]:
        solid_stream = self._index_stream(
            self.solid_count, self.target_solid_examples, self.sampler_seed,
        )
        gas_stream = self._index_stream(
            self.gas_count, self.target_gas_examples, self.sampler_seed + 1,
        )
        batch_rng = np.random.default_rng(self.sampler_seed + 2)
        for _ in range(self.global_batch_count):
            batch = [next(solid_stream) for _ in range(self.solid_per_batch)]
            batch.extend(
                self.solid_count + next(gas_stream)
                for _ in range(self.gas_per_batch)
            )
            batch_rng.shuffle(batch)
            yield [int(index) for index in batch]

    def audit(self, preview_batches: int = 8) -> Dict[str, object]:
        preview = [
            {
                "indices": batch,
                "phase_order": [
                    "solid" if index < self.solid_count else "gas"
                    for index in batch
                ],
            }
            for batch in self._take_preview(preview_batches)
        ]
        return {
            "sampler": self.__class__.__name__,
            "sampler_seed": self.sampler_seed,
            "solid_manifest_count": self.solid_count,
            "gas_manifest_count": self.gas_count,
            "target_solid_examples": self.target_solid_examples,
            "target_gas_examples": self.target_gas_examples,
            "solid_per_local_batch": self.solid_per_batch,
            "gas_per_local_batch": self.gas_per_batch,
            "local_batch_size": self.local_batch_size,
            "world_size": self.world_size,
            "optimizer_steps": self.optimizer_steps,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "global_batch_count_before_sharding": self.global_batch_count,
            "local_batches_per_rank": self.local_batches_per_rank,
            "preview": preview,
        }

    def _take_preview(self, count: int) -> Sequence[List[int]]:
        if count <= 0:
            return []
        iterator = iter(self)
        return [next(iterator) for _ in range(min(int(count), len(self)))]


class RecoveryBatchSampler(Sampler[List[int]]):
    """Replay skipped pre-checkpoint batches, then continue the original stream."""

    def __init__(self, sampler, consumed_batches, skipped_batches, stop_after=None):
        self.sampler = sampler
        self.consumed_batches = int(consumed_batches)
        self.skipped_batches = tuple(int(value) for value in skipped_batches)
        if list(self.skipped_batches) != sorted(set(self.skipped_batches)):
            raise ValueError("Skipped batches must be sorted and unique")
        if not 0 <= self.consumed_batches <= sampler.local_batches_per_rank:
            raise ValueError("Invalid consumed batch cursor")
        if any(not 0 <= value < self.consumed_batches for value in self.skipped_batches):
            raise ValueError("Replay batches must precede the checkpoint cursor")
        self.completed_updates = self.consumed_batches - len(self.skipped_batches)
        available = sampler.local_batches_per_rank - self.completed_updates
        self.remaining_updates = available if stop_after is None else int(stop_after)
        if not 0 < self.remaining_updates <= available:
            raise ValueError("Invalid recovery length")

    def __len__(self):
        return self.remaining_updates * self.sampler.world_size

    def __iter__(self):
        skipped = set(self.skipped_batches)
        emitted = 0
        for index, batch in enumerate(self.sampler):
            original_step = index // self.sampler.world_size
            if original_step not in skipped and original_step < self.consumed_batches:
                continue
            yield batch
            emitted += 1
            if emitted == len(self):
                return
        if emitted != len(self):
            raise RuntimeError("Recovery sampler exhausted unexpectedly")


class ValidatedRecoveryDataset(Dataset):
    """Replace invalid labels with auditable same-phase draws only during recovery."""

    def __init__(self, dataset, seed, max_replacements=32):
        self.dataset = dataset
        self.seed = int(seed)
        self.max_replacements = int(max_replacements)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        requested = int(index)
        is_solid = requested < self.dataset.solid_size
        offset = 0 if is_solid else self.dataset.solid_size
        count = self.dataset.solid_size if is_solid else self.dataset.gas_size
        rejected = []
        candidate = requested
        for attempt in range(self.max_replacements + 1):
            sample, info = self.dataset[candidate]
            invalid = [
                key for key, value in sample.items()
                if isinstance(value, torch.Tensor) and value.is_floating_point()
                and not bool(torch.isfinite(value).all())
            ]
            if any(key not in {"points_tgt", "F", "C", "gas_advection_displacement"}
                   for key in invalid):
                raise FloatingPointError(
                    f"Invalid input in {sample['model']}: {invalid}"
                )
            if not invalid:
                sample = dict(sample)
                sample["recovery_requested_index"] = requested
                sample["recovery_actual_index"] = candidate
                sample["recovery_rejections"] = json.dumps(rejected, sort_keys=True)
                return sample, info
            rejected.append({"index": candidate, "sample": sample["model"], "fields": invalid})
            key = f"{self.seed}:{requested}:{attempt}".encode("ascii")
            candidate = offset + int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % count
        raise FloatingPointError(f"Replacement budget exhausted: {rejected}")
