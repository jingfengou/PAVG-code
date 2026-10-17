# Verify your setup

Run these checks from the repository root after installation.

## Source and dependencies

```bash
python scripts/audit_release.py
python scripts/check_dependencies.py
```

The source audit checks Python syntax, machine-specific paths, credential
patterns, generated artifacts and documentation links. The dependency check
verifies the pinned PhysCtrl source and the PAVG extensions.

## Prepare an input on CPU

```bash
python scripts/run.py predict -- \
  --config configs/stage2.yaml \
  --checkpoint weights/pavg/model.safetensors \
  --phase-inputs examples/interaction/phase_inputs.npz \
  --condition-input examples/interaction/condition_input.npz \
  --output outputs/check/prepared.npz \
  --normalization joint --prepare-only
```

This writes adapted model inputs and a JSON settings sidecar. In
`--prepare-only` mode, the checkpoint argument is a placeholder; no weights
are opened. Check that the two phase slots remain in `[gas, solid]` order and
share one normalization transform.

## Check projection and tracking media

```bash
python scripts/run.py project -- \
  --phase-inputs examples/gas/phase_inputs.npz \
  --gas-mask examples/gas/gas_mask.png \
  --demo --output-dir outputs/check/tracks
python scripts/run.py render-tracks -- \
  --tracks outputs/check/tracks/tracks.npy \
  --output-dir outputs/check/tracking --write-mp4
python scripts/run.py video -- \
  --input-image examples/gas/input.png \
  --tracking-video outputs/check/tracking/tracks_tracking.mp4 \
  --checkpoint weights/das \
  --output-dir outputs/check/video \
  --prompt "A cloud of smoke drifts upward and to the left." \
  --check-only
```

`--demo` uses synthetic motion to exercise projection and rendering.
`--check-only` decodes the tracking video and checks its 49 frames, 24 fps and
720 x 480 resolution without loading video weights.

## Evaluate a model

Use a matching checkpoint and configuration for model inference. The CPU
checks above cover input and media interfaces; assessing prediction quality
requires the simulation targets and evaluation commands described in
[Data and Evaluation](DATA_AND_EVALUATION.md). Use the intended Linux/CUDA
environment for training and full video generation.
