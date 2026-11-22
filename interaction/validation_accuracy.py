"""Full, fixed-seed validation and protected best-model selection."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from diffusers import DDIMScheduler
from safetensors.torch import save_model
from torch.utils.data import default_collate

from metrics import macro_average, metric_protocol, sequence_metrics
from split_phase_exposure import select_phase


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def summarize(records, expected_groups):
    records = sorted(records, key=lambda r: r["index"])
    if [r["index"] for r in records] != list(range(sum(expected_groups.values()))):
        raise ValueError("Validation indices are incomplete or duplicated")
    counts = Counter(r["group"] for r in records)
    if dict(counts) != dict(expected_groups):
        raise ValueError(f"Validation group coverage differs: {dict(counts)}")
    groups = {name: macro_average([r["metrics"] for r in records if r["group"] == name])
              for name in ("solid", "gas", "interaction")}
    overall = {key: 0.25 * groups["solid"][key] + 0.25 * groups["gas"][key]
               + 0.50 * groups["interaction"][key] for key in groups["solid"]}
    if not all(np.isfinite(v) for group in [*groups.values(), overall] for v in group.values()):
        raise FloatingPointError("Non-finite validation metrics")
    return {"counts": dict(counts), "groups": groups, "overall": overall,
            "selection_key": [overall["vIoU"], -overall["CD"], -overall["Corr"]]}


@torch.inference_mode()
def predict_phase(model, scheduler, data, slot, steps, seed, device):
    batch = {key: value.to(device) for key, value in
             default_collate([select_phase(data, slot)]).items()}
    target = batch["points_tgt"]
    generator = torch.Generator(device=device).manual_seed(seed)
    sample = torch.randn(target.shape, device=device, generator=generator)
    presence = batch["phase_presence"]
    sample *= presence[:, None, :, None, None]
    scheduler.set_timesteps(steps, device=device)
    for timestep in scheduler.timesteps:
        times = torch.full((1,), int(timestep), device=device, dtype=torch.long)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(
                sample, times, batch["points_src"], batch["init_vel"], batch["force"],
                batch["drag_point"], batch["drag_mask"],
                **{key: batch[key] for key in (
                    "physics_E", "physics_nu", "physics_mat_type", "physics_floor_height",
                    "physics_gravity", "physics_coeff", "physics_volume", "phase_presence"
                )},
            )
        sample = scheduler.step(output.float(), timestep, sample).prev_sample
        sample *= presence[:, None, :, None, None]
    prediction = sample[0, :, slot].float().cpu().numpy()
    target = target[0, :, slot].float().cpu().numpy()
    if not np.isfinite(prediction).all() or not np.isfinite(target).all():
        raise FloatingPointError("Non-finite full-validation prediction or target")
    return prediction, target


def validate_accuracy(model, dataset, cfg, accelerator, step, output_dir, select_best=True):
    """Shard every original val index exactly once; never read test samples."""
    started = time.monotonic()
    settings = cfg.validation_accuracy
    expected = dict(settings.expected_groups)
    manifest = json.loads(Path(str(cfg.val_list)).read_text())
    if len(dataset) != sum(expected.values()) or dataset.files != manifest:
        raise ValueError("Validation dataset differs from the frozen complete manifest")
    raw = accelerator.unwrap_model(model)
    was_training = raw.training
    np_state, py_state = np.random.get_state(), random.getstate()
    records = []
    error = None
    raw.eval()
    try:
        with torch.random.fork_rng(devices=[accelerator.device.index]
                                   if accelerator.device.type == "cuda" else []):
            scheduler = DDIMScheduler(
                num_train_timesteps=1000, prediction_type="sample", clip_sample=False
            )
            for index in range(accelerator.process_index, len(dataset), accelerator.num_processes):
                seed = int(settings.seed) + index
                torch.manual_seed(seed)
                np.random.seed(seed)
                random.seed(seed)
                data, meta = dataset[index]
                if not bool(data["sample_valid"]) or any(
                    value.is_floating_point() and not bool(torch.isfinite(value).all())
                    for value in data.values()
                ):
                    raise FloatingPointError(f"Invalid validation data: {meta['file']}")
                predictions, targets, streams = [], [], {}
                for name, slot in (("gas", 0), ("solid", 1)):
                    if bool(data["phase_presence"][slot]):
                        pred, target = predict_phase(
                            raw, scheduler, data, slot, int(settings.ddim_steps),
                            seed, accelerator.device,
                        )
                        streams[name] = sequence_metrics(pred, target)
                        predictions.append(pred)
                        targets.append(target)
                combined_pred = np.concatenate(predictions, axis=1)
                combined_target = np.concatenate(targets, axis=1)
                group = manifest[index]["phase"]
                records.append({
                    "index": index, "source": str(meta["file"]), "group": group,
                    "seed": seed, "metrics": (streams[group] if group != "interaction" else
                                             sequence_metrics(combined_pred, combined_target)),
                    "stream_metrics": streams,
                    "prediction_sha256": hashlib.sha256(combined_pred.tobytes()).hexdigest(),
                    "target_sha256": hashlib.sha256(combined_target.tobytes()).hexdigest(),
                })
                if len(records) % 10 == 0:
                    write_json(Path(str(cfg.get("runtime_dir", output_dir))) /
                               f"validation_progress_rank{accelerator.process_index}.json", {
                                   "step": step, "done": len(records), "last_index": index,
                                   "elapsed_seconds": time.monotonic() - started,
                               })
    except Exception as exc:
        error = repr(exc)
    finally:
        raw.train(was_training)
        np.random.set_state(np_state)
        random.setstate(py_state)
    packets = [None] * accelerator.num_processes
    if accelerator.num_processes > 1:
        dist.all_gather_object(packets, {"records": records, "error": error})
    else:
        packets[0] = {"records": records, "error": error}
    errors = [p["error"] for p in packets if p["error"]]
    if errors:
        raise RuntimeError(f"Full validation failed: {errors}")
    records = [r for packet in packets for r in packet["records"]]
    summary = summarize(records, expected)
    summary.update(
        step=int(step), seed=int(settings.seed), ddim_steps=int(settings.ddim_steps),
        elapsed_seconds=time.monotonic() - started,
        completed_at_utc=datetime.now(timezone.utc).isoformat(),
        val_manifest_sha256=sha(cfg.val_list), metric_protocol=metric_protocol(),
        aggregation="0.25 solid + 0.25 gas + 0.50 interaction",
        config_sha256=sha(cfg.run_config_path),
        model_variant="shared predictor with independent phase samples",
    )
    if accelerator.is_main_process:
        destination = Path(output_dir)
        write_json(destination / "validation" / f"step_{step:06d}.json",
                   {**summary, "per_sample": sorted(records, key=lambda r: r["index"])})
        best_path = destination / "best" / "selection.json"
        old = json.loads(best_path.read_text()) if best_path.exists() else None
        improved = old is None or tuple(summary["selection_key"]) > tuple(old["selection_key"])
        if select_best and improved:
            best_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint = best_path.parent / "model.safetensors"
            temporary = checkpoint.with_suffix(".safetensors.partial")
            save_model(raw, str(temporary))
            checkpoint_hash = sha(temporary)
            temporary.replace(checkpoint)
            write_json(best_path, {**summary, "checkpoint": str(checkpoint),
                                   "checkpoint_sha256": checkpoint_hash,
                                   "rule": "maximize Overall vIoU; ties minimize CD then Corr"})
        write_json(destination / "validation" / "latest.json",
                   {**summary, "improved": improved, "selected": select_best and improved})
    accelerator.wait_for_everyone()
    return summary
