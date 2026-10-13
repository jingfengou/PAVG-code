"""Create local configs without embedding machine paths in tracked source."""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]


def create_configs(data_root, manifest_root, run_root, metadata_sidecar, output_dir):
    data_root = str(Path(data_root).resolve())
    manifest_root = Path(manifest_root).resolve()
    run_root = Path(run_root).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for phase in ("solid", "gas", "unified"):
        cfg = OmegaConf.load(ROOT / "configs/stage1.yaml")
        cfg.output_dir = str(run_root / f"stage1_{phase}")
        cfg.tracker_project_name = f"pact_stage1_{phase}"
        cfg.train_dataset.dataset_path = data_root
        cfg.train_dataset.category = f"pact_{phase}"
        cfg.train_dataset.dataset_list = str(manifest_root / f"{phase}_train_paths.json")
        cfg.val_dataset_list = str(manifest_root / f"{phase}_val_paths.json")
        cfg.test_dataset_list = str(manifest_root / f"{phase}_test_paths.json")
        cfg.lambda_gas_divergence = 0.05 if phase != "solid" else 0.0
        if phase == "gas":
            cfg.seed = cfg.train_dataset.seed = 20260820
            cfg.train_batch_size = 2
            cfg.gradient_accumulation_steps = 2
            cfg.train_dataset.has_gravity = False
            cfg.lambda_deform = cfg.lambda_floor = 0.0
        elif phase == "unified":
            cfg.max_train_steps = 40000
            cfg.train_batch_size = 4
            cfg.gradient_accumulation_steps = 1
            cfg.phase_balanced = {
                "enabled": True,
                "expected_world_size": 24,
                "local_batch_solid": 2,
                "local_batch_gas": 2,
                "target_exposure_per_phase": 1920000,
                "sampler_seed": 20260822,
                "solid_dataset": {
                    "dataset_list": str(manifest_root / "solid_train_paths.json"),
                    "has_gravity": True,
                },
                "gas_dataset": {
                    "dataset_list": str(manifest_root / "gas_train_paths.json"),
                    "has_gravity": False,
                    "seed": 20260820,
                },
            }
        target = output_dir / f"stage1_{phase}.yaml"
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite {target}")
        OmegaConf.save(cfg, target)
    cfg = OmegaConf.load(ROOT / "configs/stage2.yaml")
    cfg.mdm_root = str(ROOT / "src")
    cfg.dataset_root = data_root
    cfg.output_dir = str(run_root / "stage2")
    cfg.aoss_cache_dir = str(run_root / "cache")
    cfg.object_metadata_sidecar = (
        str(Path(metadata_sidecar).resolve()) if metadata_sidecar else ""
    )
    for split in ("train", "val", "test"):
        cfg[f"{split}_list"] = str(manifest_root / f"combined_{split}_paths.json")
    for phase in ("solid", "gas"):
        cfg.initialization[f"{phase}_checkpoint"] = str(
            run_root / f"stage1_{phase}/checkpoint-60000/model.safetensors"
        )
    target = output_dir / "stage2.yaml"
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite {target}")
    OmegaConf.save(cfg, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest-root", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--metadata-sidecar")
    parser.add_argument("--output-dir", default="configs/local")
    args = parser.parse_args()
    create_configs(**vars(args))


if __name__ == "__main__":
    main()
