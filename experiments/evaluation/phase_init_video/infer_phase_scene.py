"""Run a prepared image-phase point cloud through the Stage 2 checkpoint."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import sys
import subprocess
import time
from pathlib import Path

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "interaction"))

from joint_normalization import normalize_scene, restore_prediction


CONDITION_KEYS = (
    "init_vel",
    "force",
    "drag_point",
    "drag_mask",
    "physics_E",
    "physics_nu",
    "physics_mat_type",
    "physics_floor_height",
    "physics_gravity",
    "physics_coeff",
    "physics_volume",
)


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_model_runtime():
    mdm_root = ROOT / "src"
    sys.path.insert(0, str(mdm_root))
    mdm_model = _load_module(
        mdm_root / "model" / "spacetime.py", "phase_scene_mdm_spacetime"
    )
    interaction_spacetime = _load_module(
        ROOT / "interaction" / "model" / "spacetime_interaction.py",
        "phase_scene_spacetime_interaction",
    )
    interaction_spacetime.configure(mdm_model)
    model_module = _load_module(
        ROOT / "interaction" / "model" / "joint_model.py",
        "phase_scene_joint_model",
    )
    return interaction_spacetime, model_module


def _unit_vector(value: list[float]) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-8 else vector


def _presence(mode: str, stored: np.ndarray) -> np.ndarray:
    if mode == "dataset":
        return stored.astype(bool)
    return {
        "gas": np.asarray([True, False]),
        "solid": np.asarray([False, True]),
        "interaction": np.asarray([True, True]),
    }[mode]


def _stored_array(source, key: str, default: np.ndarray) -> tuple[np.ndarray, bool]:
    if key in source.files:
        value = np.asarray(source[key])
        if value.shape != default.shape:
            raise ValueError(
                f"Stored {key} has shape {value.shape}, expected {default.shape}"
            )
        return value.astype(default.dtype, copy=False), True
    return default, False


def _mask_inactive_conditions(
    phase_presence: np.ndarray,
    arrays: dict[str, np.ndarray],
) -> None:
    inactive = ~phase_presence
    for key in (
        "init_vel",
        "force",
        "drag_point",
        "drag_mask",
        "physics_E",
        "physics_nu",
        "physics_floor_height",
        "physics_gravity",
        "physics_coeff",
        "physics_volume",
    ):
        arrays[key][inactive] = 0
    arrays["physics_mat_type"][inactive] = 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--phase-inputs", required=True, type=Path)
    parser.add_argument(
        "--condition-input",
        type=Path,
        help="Optional NPZ supplying the physical-condition tensors.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--phase-mode",
        choices=["dataset", "gas", "solid", "interaction"],
        default="dataset",
    )
    parser.add_argument("--solid-force", type=float, nargs=3, default=(0.0, 1.0, 0.0))
    parser.add_argument("--solid-force-magnitude", type=float, default=1.0)
    parser.add_argument("--solid-force-coeff", type=float, default=0.02)
    parser.add_argument("--solid-log-e", type=float, default=4.0)
    parser.add_argument("--solid-nu", type=float, default=0.4)
    parser.add_argument("--solid-drag-radius", type=float, default=0.25)
    parser.add_argument("--gas-velocity", type=float, nargs=3, default=(0.08, 0.16, 0.0))
    parser.add_argument("--num-inference-steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--normalization", choices=["joint", "none"], default="joint",
                        help="Shared bounding-box adaptation; none selects the legacy interface.")
    parser.add_argument("--interaction", choices=["on", "off"], default="on")
    parser.add_argument("--expected-checkpoint-sha256", help="Reject an unexpected checkpoint hash.")
    parser.add_argument("--prepare-only", action="store_true",
                        help="Save adapted inputs on CPU without importing models or requiring weights.")
    args = parser.parse_args()
    started_at_utc = dt.datetime.now(dt.timezone.utc).isoformat()

    config_values = yaml.safe_load(args.config.read_text(encoding="utf-8-sig"))
    point_size = int(config_values["pc_size"])
    if int(config_values["n_training_frames"]) != 24:
        raise ValueError("This scene-video protocol requires 24 future trajectory states")

    source = np.load(args.phase_inputs, allow_pickle=False)
    condition_source = (
        np.load(args.condition_input, allow_pickle=False)
        if args.condition_input is not None
        else source
    )
    point_key = "points_src" if "points_src" in source else "init_pc"
    points_src = np.asarray(source[point_key], dtype=np.float32)
    if points_src.shape != (2, point_size, 3):
        raise ValueError(f"Unexpected points_src shape: {points_src.shape}")
    stored_presence = np.asarray(source["phase_presence"], dtype=bool)
    if stored_presence.shape != (2,) or not stored_presence.any():
        raise ValueError("Expected active phase_presence in [gas, solid] order")
    phase_presence = _presence(
        args.phase_mode, stored_presence
    )
    if np.any(phase_presence & ~stored_presence):
        raise ValueError("A requested phase is absent from the prepared input")
    points_src = points_src.copy()
    original_points = points_src.copy()

    point_count = points_src.shape[1]
    gas_velocity = np.asarray(args.gas_velocity, dtype=np.float32)
    solid_force = (
        _unit_vector(list(args.solid_force)) * float(args.solid_force_magnitude)
    ).astype(np.float32)
    default_init_vel = np.zeros_like(points_src)
    default_init_vel[0] = gas_velocity
    default_force = np.stack([gas_velocity, solid_force], axis=0).astype(np.float32)

    solid_drag_index = int(np.argmax(points_src[1, :, 0]))
    solid_drag_point = points_src[1, solid_drag_index]
    solid_distance = np.max(np.abs(points_src[1] - solid_drag_point), axis=-1)
    solid_drag_mask = solid_distance < float(args.solid_drag_radius)
    default_drag_point = np.stack(
        [np.zeros(3, dtype=np.float32), solid_drag_point], axis=0
    )
    default_drag_mask = np.stack(
        [np.ones(point_count, dtype=np.float32), solid_drag_mask.astype(np.float32)],
        axis=0,
    )[..., None]

    defaults = {
        "init_vel": default_init_vel,
        "force": default_force,
        "drag_point": default_drag_point,
        "drag_mask": default_drag_mask,
        "physics_E": np.asarray([[0.0], [args.solid_log_e]], dtype=np.float32),
        "physics_nu": np.asarray([[0.0], [args.solid_nu]], dtype=np.float32),
        "physics_mat_type": np.asarray([4, 0], dtype=np.int64),
        "physics_floor_height": np.asarray([[-2.5], [-2.5]], dtype=np.float32),
        "physics_gravity": np.asarray([0, 0], dtype=np.int64),
        "physics_coeff": np.asarray(
            [[1.0], [args.solid_force_coeff]], dtype=np.float32
        ),
        "physics_volume": np.ones((2, point_count, 1), dtype=np.float32),
    }
    conditions: dict[str, np.ndarray] = {}
    stored_conditions = []
    for key in CONDITION_KEYS:
        conditions[key], stored = _stored_array(
            condition_source, key, defaults[key]
        )
        if stored:
            stored_conditions.append(key)
    # Fit once to all originally active phases. Gas-only and interaction-OFF
    # controls keep the two-phase transform rather than moving the gas again.
    original = {"init_pc": points_src, "phase_presence": stored_presence, **conditions}
    if args.normalization == "joint":
        adapted, center, scale = normalize_scene(original)
    else:
        adapted = {key: value.copy() for key, value in original.items()}
        center, scale = np.zeros(3, dtype=np.float32), 1.0
    adapted["phase_presence"] = phase_presence.copy()
    adapted["init_pc"][~phase_presence] = 0
    points_src = adapted["init_pc"]
    model_conditions = {key: adapted[key] for key in CONDITION_KEYS}
    _mask_inactive_conditions(phase_presence, model_conditions)
    _mask_inactive_conditions(phase_presence, conditions)
    original_points[~phase_presence] = 0
    if not all(np.isfinite(value).all() for value in adapted.values()):
        raise ValueError("Prepared model inputs contain non-finite values")
    preparation = {
        "protocol_id": "pavg-scene-video-v1",
        "normalization": args.normalization,
        "center": center.tolist(), "scale": scale,
        "phase_presence": phase_presence.tolist(),
        "original_phase_presence": stored_presence.tolist(),
        "cross_attention_enabled": args.interaction == "on",
        "trajectory_seed": args.seed, "DDIM_steps": args.num_inference_steps,
        "CFG": args.guidance_scale, "generator": "CPU",
        "future_states": 24, "video_frames": 49, "video_fps": 24,
        "first_to_last_seconds": 2.0,
        "started_at_utc": started_at_utc,
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "phase_inputs_sha256": hashlib.sha256(args.phase_inputs.read_bytes()).hexdigest(),
        "condition_input_sha256": (hashlib.sha256(args.condition_input.read_bytes()).hexdigest()
                                   if args.condition_input else None),
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "normalization_source_sha256": hashlib.sha256(
            Path(__file__).with_name("joint_normalization.py").read_bytes()).hexdigest(),
    }
    commit_query = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                  capture_output=True, text=True)
    preparation["code_commit"] = commit_query.stdout.strip() if commit_query.returncode == 0 else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.prepare_only:
        np.savez_compressed(args.output, **adapted, original_init_pc=original_points,
                            normalization_center=center, normalization_scale=scale)
        preparation["status"] = "prepared_only_no_model_inference"
        preparation["finished_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        args.output.with_suffix(".json").write_text(json.dumps(preparation, indent=2), encoding="utf-8")
        print(json.dumps(preparation, indent=2))
        return

    import torch
    from diffusers import DDIMScheduler
    from omegaconf import OmegaConf
    from safetensors.torch import load_file
    from model.joint_pipeline import JointTrajPipeline

    cfg = OmegaConf.load(args.config)
    checkpoint = args.checkpoint
    if checkpoint.is_dir():
        checkpoint = checkpoint / "model.safetensors"
    checksum = hashlib.sha256()
    with checkpoint.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            checksum.update(chunk)
    checkpoint_sha256 = checksum.hexdigest()
    if args.expected_checkpoint_sha256 and checkpoint_sha256 != args.expected_checkpoint_sha256:
        raise ValueError("Checkpoint SHA256 differs from the requested inference protocol")
    mdm_model, model_mod = _load_model_runtime()
    model = model_mod.JointMDM_ST(
        int(cfg.pc_size), int(cfg.n_training_frames), cfg.model_config, mdm_model
    )
    model.load_state_dict(load_file(str(checkpoint), device="cpu"), strict=True)
    model.set_cross_attention_enabled(args.interaction == "on")

    device = torch.device(args.device)
    tensor = lambda value: torch.from_numpy(value).unsqueeze(0).to(device)
    pipeline = JointTrajPipeline(
        model,
        DDIMScheduler(
            num_train_timesteps=1000,
            prediction_type="sample",
            clip_sample=False,
        ),
    )
    autocast_enabled = device.type == "cuda"
    if autocast_enabled:
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=autocast_enabled,
    ):
        prediction = pipeline(
            tensor(points_src),
            tensor(model_conditions["init_vel"]),
            tensor(model_conditions["force"]),
            tensor(model_conditions["drag_point"]),
            tensor(model_conditions["drag_mask"]),
            tensor(model_conditions["physics_E"]),
            tensor(model_conditions["physics_nu"]),
            tensor(model_conditions["physics_mat_type"]),
            tensor(model_conditions["physics_floor_height"]),
            tensor(model_conditions["physics_gravity"]),
            tensor(model_conditions["physics_coeff"]),
            tensor(model_conditions["physics_volume"]),
            tensor(phase_presence),
            device=device,
            generator=torch.Generator(device="cpu").manual_seed(args.seed),
            n_frames=int(cfg.n_training_frames),
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
        )
    elapsed_seconds = time.perf_counter() - started
    prediction_np = prediction[0].float().cpu().numpy()
    if not bool(np.isfinite(prediction_np).all()):
        raise RuntimeError("Stage 2 prediction contains non-finite values")
    restored = restore_prediction(prediction_np, phase_presence, center, scale)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        prediction=restored,
        restored_prediction=restored,
        prediction_normalized=prediction_np,
        init_pc=original_points,
        model_init_pc=points_src,
        normalization_center=center,
        normalization_scale=scale,
        phase_presence=phase_presence,
        **conditions,
    )
    displacement = np.linalg.norm(restored[-1] - original_points, axis=-1).mean(axis=-1)
    report = {
        **preparation,
        "status": "complete",
        "finished_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "strict_load": True,
        "precision": "bfloat16" if autocast_enabled else "float32",
        "prediction_coordinates": "original_scene",
        "phase_mode": args.phase_mode,
        "phase_presence": phase_presence.tolist(),
        "prediction_shape": list(prediction_np.shape),
        "conditioning": (
            "stored"
            if len(stored_conditions) == len(CONDITION_KEYS)
            else "defaults"
            if not stored_conditions
            else "mixed"
        ),
        "stored_condition_keys": stored_conditions,
        "condition_input": str(args.condition_input) if args.condition_input else None,
        "elapsed_seconds": elapsed_seconds,
        "peak_cuda_memory_gib": (
            float(torch.cuda.max_memory_allocated(device) / (1024**3))
            if autocast_enabled
            else 0.0
        ),
        "mean_final_displacement": {
            "gas": float(displacement[0]) if phase_presence[0] else None,
            "solid": float(displacement[1]) if phase_presence[1] else None,
        },
        "solid_drag_points": int((conditions["drag_mask"][1] > 0).sum()),
        "output": str(args.output),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
