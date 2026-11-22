"""A frozen mixed ordering of the three original training exposure budgets."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def make_plan(source_sizes, exposures, seed):
    rng = np.random.default_rng(seed)
    pieces, offset = [], 0
    for size, count in zip(source_sizes, exposures):
        remaining, cycles = int(count), []
        while remaining:
            permutation = rng.permutation(size)[:remaining]
            cycles.append(permutation + offset)
            remaining -= len(permutation)
        pieces.append(np.concatenate(cycles))
        offset += size
    plan = np.concatenate(pieces).astype(np.int64)
    rng.shuffle(plan)
    return plan


class ExposurePlanDataset(Dataset):
    def __init__(self, dataset, settings, output_dir):
        self.dataset = dataset
        self.settings = settings
        self.plan = np.load(settings["path"], mmap_mode="r")
        self.source_ids = np.asarray([r["exposure_source"] for r in dataset.files], dtype=np.int8)
        self.pools = {}
        for index, row in enumerate(dataset.files):
            self.pools.setdefault((row["exposure_source"], row["phase"]), []).append(index)
        self.output_dir = Path(output_dir)
        self.seed = int(settings["replacement_seed"])
        self._prefix_counts = {}

    def __len__(self):
        return len(self.plan)

    def expected_counts(self, steps, world_size, batch_size, rank):
        key = (world_size, batch_size, rank)
        if key not in self._prefix_counts:
            sources = self.source_ids[self.plan].reshape(-1, world_size, batch_size)[:, rank]
            per_step = np.stack([(sources == s).sum(axis=1) for s in range(3)], axis=1)
            self._prefix_counts[key] = np.concatenate(
                [np.zeros((1, 3), dtype=np.int64), per_step.cumsum(axis=0)], axis=0
            )
        return self._prefix_counts[key][int(steps)].tolist()

    def __getitem__(self, position):
        requested = int(self.plan[position])
        row = self.dataset.files[requested]
        candidate, rejected = requested, []
        pool = self.pools[(row["exposure_source"], row["phase"])]
        for attempt in range(33):
            data, meta = self.dataset[candidate]
            invalid = [
                key for key, value in data.items()
                if isinstance(value, torch.Tensor) and value.is_floating_point()
                and not bool(torch.isfinite(value).all())
            ]
            label_fields = {
                "points_tgt", "physics_F", "physics_C", "gas_target_velocity",
                "gas_advection_displacement",
            }
            if set(invalid) - label_fields:
                raise FloatingPointError(f"Invalid input fields in {meta['file']}: {invalid}")
            if not invalid and bool(data["sample_valid"]):
                meta = {**meta, "exposure_source": row["exposure_source"],
                        "exposure_position": int(position)}
                if rejected:
                    self.output_dir.mkdir(parents=True, exist_ok=True)
                    record = {"position": int(position), "source": row["exposure_source"],
                              "requested": requested, "actual": candidate,
                              "actual_file": meta["file"], "rejections": rejected}
                    audit = self.output_dir / (
                        f"replacement_rank{os.environ.get('RANK', '0')}_pid{os.getpid()}.jsonl"
                    )
                    with audit.open("a") as handle:
                        handle.write(json.dumps(record) + "\n")
                return data, meta
            if not invalid:
                raise FloatingPointError(f"Invalid sample without exposed bad-label fields: {meta['file']}")
            rejected.append({"index": candidate, "file": meta["file"], "fields": invalid})
            digest = hashlib.sha256(f"{self.seed}:{requested}:{attempt}".encode()).digest()
            candidate = pool[int.from_bytes(digest[:8], "big") % len(pool)]
        raise FloatingPointError(f"Replacement budget exhausted: {rejected}")
