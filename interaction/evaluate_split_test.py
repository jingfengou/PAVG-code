"""Evaluate a validation-selected split-phase checkpoint on the frozen test set."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from diffusers import DDIMScheduler
from omegaconf import OmegaConf
from safetensors.torch import load_file, load_model

from evaluate_dual_joint_fixed_20260905 import _sample_metrics
from metrics import macro_average, metric_protocol
from train_interaction import build_model, configure_attention_backend, load_runtime
from validation_accuracy import predict_phase, sha, write_json


COUNTS = {"solid": 500, "gas": 100, "interaction": 100}


def utc():
    return datetime.now(timezone.utc).isoformat()


def shard(args):
    started = time.monotonic()
    if sha(args.checkpoint) != args.checkpoint_sha:
        raise ValueError("Selected checkpoint hash mismatch")
    cfg = OmegaConf.load(args.config)
    if not (
        cfg.model_config.backbone_mode == "shared"
        and cfg.model_config.independent_phase_forward
        and cfg.model_config.gas_conditioning == "large"
        and cfg.model_config.object_conditioning == "large"
    ):
        raise ValueError("Expected the independently trained split-phase model")
    if args.manifest_sha and sha(cfg.test_list) != args.manifest_sha:
        raise ValueError("Frozen test manifest changed")
    manifest = json.loads(Path(cfg.test_list).read_text())
    val = json.loads(Path(cfg.val_list).read_text())
    paths = [r["path"] for r in manifest]
    if len(set(paths)) != 700 or Counter(r["phase"] for r in manifest) != COUNTS:
        raise ValueError("Unexpected test coverage")
    if set(paths) & {r["path"] for r in val}:
        raise ValueError("Test and validation overlap")
    indices = list(range(args.rank, 700, args.world_size))
    folder = args.output / f"shard_{args.rank:02d}"
    folder.mkdir(parents=True, exist_ok=False)
    configure_attention_backend("math")
    torch.set_num_threads(4)
    mdm, dataset_module, model_module = load_runtime(cfg)
    dataset = dataset_module.ObjectGasDataset("test", cfg)
    if dataset.files != manifest:
        raise ValueError("Dataset does not match the test manifest")
    device = torch.device(args.device)
    model = build_model(cfg, mdm, model_module).to(device)
    load_model(model, str(args.checkpoint), strict=True, device="cpu")
    stored = load_file(str(args.checkpoint), device="cpu")
    actual = model.state_dict()
    if not all(
        name in actual and torch.equal(actual[name].detach().cpu(), value)
        for name, value in stored.items()
    ):
        raise ValueError("Checkpoint tensors were not loaded exactly")
    del stored, actual
    model.eval().requires_grad_(False)
    scheduler = DDIMScheduler(
        num_train_timesteps=1000, prediction_type="sample", clip_sample=False
    )
    provenance = {
        "status": "running", "started_at_utc": utc(), "split": "test",
        "evaluator_sha256": sha(__file__),
        "checkpoint": str(args.checkpoint), "checkpoint_sha256": args.checkpoint_sha,
        "config": str(args.config), "config_sha256": sha(args.config),
        "manifest": str(cfg.test_list), "manifest_sha256": sha(cfg.test_list),
        "data_root": str(cfg.dataset_root), "output_dir": str(folder),
        "rank": args.rank, "world_size": args.world_size, "indices": indices,
        "seed": args.seed, "seed_policy": "seed_plus_manifest_index",
        "ddim_steps": 25, "attention_backend": "math",
        "precision": "bf16_autocast_fp32_scheduler" if device.type == "cuda" else "float32",
        "inference_protocol": "independent_2048_per_phase_no_joint_attention",
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "gpu_uuid": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "metric_protocol": metric_protocol(),
    }
    write_json(folder / "provenance.json", provenance)
    records = []
    with torch.inference_mode():
        for index in indices:
            seed = args.seed + index
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            data, meta = dataset[index]
            if not bool(data["sample_valid"]) or Path(str(meta["file"])).resolve() != Path(dataset.dataset_root, paths[index]).resolve():
                raise ValueError(f"Invalid test item {index}")
            if any(v.is_floating_point() and not bool(torch.isfinite(v).all())
                   for v in data.values()):
                raise FloatingPointError(f"Non-finite test data {index}")
            initial = data["points_src"].numpy()
            target = data["points_tgt"].numpy()
            prediction = np.zeros_like(target)
            streams = {}
            active = []
            for name, slot in (("gas", 0), ("solid", 1)):
                if bool(data["phase_presence"][slot]):
                    pred, gt = predict_phase(
                        model, scheduler, data, slot, 25, seed, device
                    )
                    if not np.array_equal(gt, target[:, slot]):
                        raise ValueError("Phase target changed during prediction")
                    prediction[:, slot] = pred
                    streams[name] = _sample_metrics(pred, gt, initial[slot])
                    active.append(slot)
            group = manifest[index]["phase"]
            if group == "interaction":
                streams[group] = _sample_metrics(
                    np.concatenate([prediction[:, s] for s in active], axis=1),
                    np.concatenate([target[:, s] for s in active], axis=1),
                    np.concatenate([initial[s] for s in active], axis=0),
                )
            row = {
                "index": index, "source": paths[index], "group": group,
                "phase_presence": data["phase_presence"].tolist(), "seed": seed,
                "input_file_sha256": sha(Path(dataset.dataset_root, paths[index])),
                "initial_sha256": hashlib.sha256(initial.tobytes()).hexdigest(),
                "target_sha256": hashlib.sha256(target.tobytes()).hexdigest(),
                "prediction_sha256": hashlib.sha256(prediction.tobytes()).hexdigest(),
                "metrics": streams[group], "stream_metrics": streams,
            }
            path = folder / f"sample_{index:03d}.npz"
            np.savez_compressed(
                path, prediction=prediction, target=target, initial=initial
            )
            row["prediction_file_sha256"] = sha(path)
            write_json(folder / f"sample_{index:03d}.json", row)
            records.append(row)
            progress = {
                "status": "running", "done": len(records), "total": len(indices),
                "index": index, "utc": utc(),
                "elapsed_seconds": time.monotonic() - started,
            }
            write_json(folder / "progress.json", progress)
            print(json.dumps(progress), flush=True)
    provenance.update(
        status="complete", finished_at_utc=utc(),
        elapsed_seconds=time.monotonic() - started,
    )
    write_json(folder / "result.json", {"provenance": provenance, "per_sample": records})
    write_json(folder / "provenance.json", provenance)
    write_json(folder / "progress.json", {
        "status": "complete", "done": len(records), "total": len(indices), "utc": utc()
    })


def aggregate(args):
    packets = [
        json.loads((args.output / f"shard_{r:02d}/result.json").read_text())
        for r in range(args.world_size)
    ]
    records = sorted(
        [row for p in packets for row in p["per_sample"]], key=lambda r: r["index"]
    )
    if [r["index"] for r in records] != list(range(700)):
        raise ValueError("Test samples are missing or duplicated")
    if Counter(r["group"] for r in records) != COUNTS:
        raise ValueError("Test group coverage changed")
    keys = (
        "checkpoint_sha256", "config_sha256", "manifest_sha256",
        "seed", "ddim_steps", "inference_protocol",
    )
    first = packets[0]["provenance"]
    for packet in packets:
        p = packet["provenance"]
        if p["status"] != "complete" or any(p[k] != first[k] for k in keys):
            raise ValueError("Inconsistent shard provenance")
    reference = json.loads(args.reference.read_text()) if args.reference else None
    if reference:
        previous = {r["index"]: r for r in reference["per_sample"]}
        for row in records:
            for key in ("source", "input_file_sha256", "initial_sha256", "target_sha256"):
                if row[key] != previous[row["index"]][key]:
                    raise ValueError(f"Paper input mismatch at {row['index']}: {key}")
    groups = {
        name: macro_average([r["metrics"] for r in records if r["group"] == name])
        for name in COUNTS
    }
    streams = {
        name: macro_average([
            r["stream_metrics"][name] for r in records if r["group"] == "interaction"
        ]) for name in ("solid", "gas")
    }
    common = set.intersection(*(set(g) for g in groups.values()))
    overall = {
        k: .25 * groups["solid"][k] + .25 * groups["gas"][k]
        + .5 * groups["interaction"][k] for k in sorted(common)
    }
    if not all(np.isfinite(v) for g in [*groups.values(), *streams.values(), overall]
               for v in g.values()):
        raise FloatingPointError("Non-finite aggregated metrics")
    result = {
        "status": "complete", "split": "test", "num_samples": 700,
        "counts": COUNTS, "groups": groups, "interaction_streams": streams,
        "overall": overall, "per_sample": records, "provenance": first,
        "finished_at_utc": utc(), "checkpoint_selected_using": "validation_only",
        "paper_input_hashes_verified": bool(reference),
        "reference_sha256": sha(args.reference) if args.reference else None,
        "aggregation": "0.25 solid + 0.25 gas + 0.50 concatenated interaction",
    }
    write_json(args.output / "summary.json", result)
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("per_sample", "provenance")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("shard", "aggregate"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--manifest-sha", help="Optional digest of the local test manifest")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--reference", type=Path)
    options = parser.parse_args()
    if options.action == "shard":
        if not 0 <= options.rank < options.world_size:
            parser.error("Invalid shard rank")
        if not all((options.config, options.checkpoint, options.checkpoint_sha)):
            parser.error("Shard requires config, checkpoint and checkpoint-sha")
        shard(options)
    else:
        aggregate(options)
