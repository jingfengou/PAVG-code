# Inference

The [README](../README.md#image-to-video-generation) runs a provided example.
Use the steps below to prepare your own image and motion controls.

## Prepare an image

Place your input image and binary phase masks in `inputs/`. Align each mask
with the image, with white pixels marking the corresponding phase.

```bash
python scripts/run.py initialize -- \
  --solid-mask inputs/solid_mask.png \
  --gas-mask inputs/gas_mask.png \
  --output-dir outputs/custom/init
```

For a gas-only scene, omit `--solid-mask`. For a solid-only scene, omit
`--gas-mask`. Initialization writes `phase_inputs.npz` containing point
clouds and projection information.

## Predict trajectories

Download the matching [PAVG weights](https://huggingface.co/irony2025/PAVG) from Hugging Face.

```bash
python scripts/run.py predict -- \
  --config configs/stage2.yaml \
  --checkpoint weights/pavg/model.safetensors \
  --phase-inputs outputs/custom/init/phase_inputs.npz \
  --solid-force 1 0 0 --solid-force-magnitude 1 \
  --gas-velocity -0.08 0.16 0 \
  --output outputs/custom/prediction.npz
```

| Option | Usage |
| --- | --- |
| `--solid-force X Y Z` | Direction of the solid force |
| `--solid-force-magnitude VALUE` | Solid force magnitude |
| `--solid-drag-radius VALUE` | Radius of the force application region |
| `--solid-log-e VALUE` | Base-10 logarithm of the solid's Young's modulus |
| `--solid-nu VALUE` | Solid Poisson ratio |
| `--gas-velocity X Y Z` | Initial gas velocity |
| `--condition-input FILE` | Load per-phase and per-point conditions from an NPZ file |
| `--device DEVICE` | Select the inference device, for example `cuda:0` |
| `--prepare-only` | Prepare inputs without loading model weights |

Values from `--condition-input` override the corresponding command-line
defaults. Use a condition file made for the same point cloud and point order.
The phase-slot order is `[gas, solid]`.

## Generate a video

Follow the `project`, `render-tracks` and `video` commands in the README,
using your own input image, masks, `phase_inputs.npz` and prediction file.
Set `--prompt` to describe the intended appearance and motion.

The prediction is saved as an NPZ file; the final video is written to
`result.mp4` in the selected video output directory.
