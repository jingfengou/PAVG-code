"""Build the ordered training catalogs and frozen exposure plans from local manifests."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "interaction"))
from exposure_plan import make_plan


def records(path, phase=None):
    items = json.loads(path.read_text(encoding="utf-8-sig"))
    if not items:
        raise ValueError(f"Empty manifest: {path}")
    rows = [dict(item) if isinstance(item, dict) else {"path": item} for item in items]
    for row in rows:
        if phase:
            row.setdefault("phase", phase)
        if row.get("phase") not in ("gas", "solid", "interaction") or not row.get("path"):
            raise ValueError(f"Every combined entry requires path and phase: {path}")
    return rows


def split_plan(parent, phases, batch=64):
    labels = phases[parent]
    paired = labels == 2
    copies = 1 + paired.astype(np.int64)
    offsets = copies.cumsum() - copies
    expanded = np.empty(int(copies.sum()), dtype=np.int64)
    expanded[offsets] = parent * 2 + np.minimum(labels, 1)
    expanded[offsets[paired]] = parent[paired] * 2
    expanded[offsets[paired] + 1] = parent[paired] * 2 + 1
    counts = np.bincount(expanded % 2, minlength=2)
    if not counts.all():
        raise ValueError("The split training schedule requires both phases")
    weights = len(expanded) / (2 * counts)
    np.random.default_rng(20261115).shuffle(expanded)
    return np.pad(expanded, (0, -len(expanded) % batch), constant_values=-1), counts, weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-root", type=Path, default=Path("manifests"))
    parser.add_argument("--output", type=Path, default=Path("manifests/prepared"))
    args = parser.parse_args()
    source, out = args.manifest_root, args.output
    out.mkdir(parents=True, exist_ok=False)
    def write(name, record):
        (out / name).write_text(json.dumps(record) + "\n", encoding="utf-8")
    phases = ["solid", "gas", "combined"]
    sources = [records(source / f"{name}_train_paths.json", None if i == 2 else name)
               for i, name in enumerate(phases)]
    catalog = [[dict(row, exposure_source=i) for row in rows] for i, rows in enumerate(sources)]
    heldout = set()
    for split in ("val", "test"):
        rows = records(source / f"combined_{split}_paths.json")
        paths = {r["path"] for r in rows}
        if heldout.intersection(paths):
            raise ValueError("Validation and test paths overlap")
        heldout.update(paths)
        write(f"solid_{split}.json", records(source / f"solid_{split}_paths.json", "solid"))
    shared = [r for group in catalog for r in group]
    if any(r["path"] in heldout for r in shared):
        raise ValueError("Training and held-out paths overlap")
    for name, rows in (("solid", catalog[0]), ("stage2", catalog[2]), ("shared", shared)):
        write(name + "_catalog.json", rows)
    budget = 1_920_000
    parent = make_plan([len(a) for a in sources], [budget] * 3, 20261107)
    phase_ids = np.asarray([("gas", "solid", "interaction").index(r["phase"]) for r in shared])
    split, counts, weights = split_plan(parent, phase_ids)
    plans = {"solid": make_plan([len(catalog[0])], [budget], 20261109),
             "stage2": make_plan([len(catalog[2])], [budget], 20261109),
             "parent": parent, "split": split}
    hashes = {}
    for name, plan in plans.items():
        path = out / f"{name}_plan.npy"
        np.save(path, plan)
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    write("plan_summary.json", {"sha256": hashes, "valid_gas_solid": counts.tolist(),
                                "weights_gas_solid": weights.tolist()})
    print(json.dumps(hashes, indent=2))


if __name__ == "__main__":
    main()
