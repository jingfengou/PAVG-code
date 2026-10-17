# Check your installation

Run these checks from the repository root after installation.

## Check dependencies

```bash
python scripts/check_dependencies.py
```

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

This writes prepared inputs to `outputs/check/prepared.npz`. The checkpoint
argument is a placeholder in `--prepare-only` mode; weights are not opened.

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

`--demo` creates example motion for previewing the tracking format.
`--check-only` checks the tracking video without loading video weights.
