"""Independent 2048-point phase samples derived from a frozen parent schedule."""

from __future__ import annotations

import numpy as np
import torch

from exposure_plan import ExposurePlanDataset
from batch_transition import exposure_cursor


PHASE_FIELDS = (
    "points_src", "init_vel", "force", "drag_point", "drag_mask", "gravity",
    "physics_E", "physics_nu", "physics_mat_type", "physics_floor_height",
    "physics_gravity", "physics_coeff", "physics_volume", "physics_F", "physics_C",
)


def select_phase(data, slot):
    """Remove the other phase before model input and objective construction."""
    slot = int(slot)
    if slot not in (0, 1) or not bool(data["phase_presence"][slot]):
        raise ValueError(f"Requested absent phase slot {slot}")
    result = {key: value.clone() for key, value in data.items()
              if not key.startswith("object_")}
    other = 1 - slot
    for key in PHASE_FIELDS:
        if key in result:
            result[key][other] = 0
    result["points_tgt"][:, other] = 0
    result["phase_presence"].zero_()
    result["phase_presence"][slot] = True
    for key in result:
        if key.startswith("interaction_") or (slot == 1 and key.startswith("gas_")):
            result[key].zero_()
    return result


class SplitPhaseExposureDataset(ExposurePlanDataset):
    def __init__(self, dataset, settings, output_dir, transition=None):
        super().__init__(dataset, settings, output_dir)
        self.tokens = self.plan
        self.real = np.asarray(self.tokens >= 0)
        # Reuse the existing audited invalid-label replacement policy.
        self.plan = np.maximum(self.tokens // 2, 0)
        self.phase_slots = np.asarray(self.tokens % 2, dtype=np.int8)
        self.source_ids_for_plan = self.source_ids[self.plan].copy()
        self.source_ids_for_plan[~self.real] = -1
        self.weights = [float(v) for v in settings["phase_weights"]]
        self.global_batch = int(settings["global_batch"])
        self.real_count = int(self.real.sum())
        self.tail_count = self.real_count % self.global_batch
        self._phase_prefix_counts = {}
        self.transition = transition
        assert self.real[:self.real_count].all() and not self.real[self.real_count:].any()

    def sample_cursor(self, steps):
        return exposure_cursor(steps, self.global_batch, self.transition)

    def _counts_by_update(self, values, categories, world_size, batch_size, rank):
        if self.transition:
            pivot = int(self.transition["step"])
            previous_batch = int(self.transition["global_batch"])
            assert previous_batch % world_size == 0
            prefix = pivot * previous_batch
            assert prefix % self.global_batch == 0
            head = values[:prefix].reshape(pivot, world_size, previous_batch // world_size)[:, rank]
            tail = values[prefix:].reshape(-1, world_size, batch_size)[:, rank]
            per_step = np.concatenate([
                np.stack([(part == category).sum(axis=1)
                          for category in range(categories)], axis=1)
                for part in (head, tail)
            ])
        else:
            samples = values.reshape(-1, world_size, batch_size)[:, rank]
            per_step = np.stack([(samples == category).sum(axis=1)
                                 for category in range(categories)], axis=1)
        return np.concatenate([np.zeros((1, categories), dtype=np.int64),
                               per_step.cumsum(axis=0)])

    def expected_counts(self, steps, world_size, batch_size, rank):
        key = (world_size, batch_size, rank)
        if key not in self._prefix_counts:
            self._prefix_counts[key] = self._counts_by_update(
                self.source_ids_for_plan, 3, world_size, batch_size, rank
            )
        return self._prefix_counts[key][int(steps)].tolist()

    def expected_phase_counts(self, steps, world_size, batch_size, rank):
        key = (world_size, batch_size, rank)
        if key not in self._phase_prefix_counts:
            self._phase_prefix_counts[key] = self._counts_by_update(
                np.where(self.real, self.phase_slots, -1), 2, world_size, batch_size, rank
            )
        return self._phase_prefix_counts[key][int(steps)].tolist()

    def __getitem__(self, position):
        data, meta = super().__getitem__(position)
        real = bool(self.real[position])
        slot = int(self.phase_slots[position]) if real else int(
            data["phase_presence"].nonzero()[0, 0]
        )
        data = select_phase(data, slot)
        weight = self.weights[slot] if real else 0.0
        # All ranks retain a fixed batch shape. Zero-loss fillers never count
        # as exposures, and the final real samples retain their global mean.
        if real and self.tail_count and position >= self.real_count - self.tail_count:
            weight *= self.global_batch / self.tail_count
        data["phase_loss_weight"] = torch.tensor(weight, dtype=torch.float32)
        meta.update(
            exposure_source=int(meta["exposure_source"]) if real else -1,
            phase_slot=slot, exposure_valid=real,
        )
        return data, meta
