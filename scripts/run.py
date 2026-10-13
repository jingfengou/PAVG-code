"""Portable entrypoints; run from the repository root."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
COMMANDS = {
    "train-stage1": "src/train.py",
    "train-stage2": "interaction/train_interaction.py",
    "eval-stage1": "experiments/evaluation/stage1/eval_fixed_single_phase.py",
    "eval-stage2": "experiments/evaluation/evaluate_dual_joint_fixed_batched_20260909.py",
    "merge-stage2": "experiments/evaluation/merge_stage2_shards.py",
    "infer-sample": "interaction/inference_interaction.py",
    "initialize": "experiments/evaluation/phase_init_video/initialize_phase_points.py",
    "predict": "experiments/evaluation/phase_init_video/infer_phase_scene.py",
    "project": "experiments/evaluation/phase_init_video/merge_phase_tracks.py",
    "render-tracks": "experiments/evaluation/phase_init_video/render_das_tracking.py",
    "video": "experiments/evaluation/phase_init_video/run_das_phase.py",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--accelerate-config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "src"), str(ROOT / "interaction"), str(ROOT),
         env.get("PYTHONPATH", "")]
    )
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [sys.executable]
    if args.command.startswith("train-"):
        command += ["-m", "accelerate.commands.launch"]
        if args.accelerate_config:
            command += ["--config_file", str(args.accelerate_config.resolve())]
        else:
            command += ["--num_processes", str(args.num_processes),
                        "--num_machines", "1", "--mixed_precision", "bf16"]
            if args.num_processes > 1:
                command += ["--multi_gpu"]
    command += [str(ROOT / COMMANDS[args.command])]
    command += extra
    print(" ".join(command), flush=True)
    if not args.dry_run:
        raise SystemExit(subprocess.call(command, cwd=ROOT, env=env))


if __name__ == "__main__":
    main()
