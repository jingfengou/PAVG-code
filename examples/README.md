# Input examples

`gas/` contains a gas-only input with upward/leftward initial motion.
`interaction/` contains a ball/dust input with both phase masks.
Each directory contains:

- `input.png`: input image.
- `gas_mask.png`, and `solid_mask.png` for the interaction example.
- `phase_inputs.npz`: numeric initial points and projection metadata.
- `condition_input.npz`: recorded numeric per-phase/per-point controls.

These files demonstrate the image, point-cloud and condition interfaces.
Simulation targets and model predictions are not included.

For a weight-free interface smoke test:

```bash
python scripts/run.py project -- \
  --phase-inputs examples/gas/phase_inputs.npz \
  --gas-mask examples/gas/gas_mask.png \
  --demo --output-dir outputs/demo_tracks
python scripts/run.py render-tracks -- \
  --tracks outputs/demo_tracks/tracks.npy \
  --output-dir outputs/demo_tracking --write-mp4
```

`--demo` produces synthetic motion for checking projection and rendering.
