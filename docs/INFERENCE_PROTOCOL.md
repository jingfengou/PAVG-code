# Scene-to-video inference

PAVG predicts 3D point trajectories, projects them into tracking controls, and
uses DaS to generate a video. Default settings are also provided in
[inference_protocol.json](../configs/inference_protocol.json). For evaluation
against simulation targets, see [Data and Evaluation](DATA_AND_EVALUATION.md).

## Shared input normalization

Phase order is `[gas, solid]`, with 2048 fixed points per active phase. Fit
one transform to the union of the originally active clouds:

```text
center = float32((union_min + union_max) / 2)
scale = 1 / max(union_max - union_min)
```

| Input | Adaptation |
| --- | --- |
| Initial point cloud | `(x - center) * scale` |
| Drag point | `(p - center) * scale` |
| Floor height | `(h - center_y) * scale` |
| Initial velocity | `v * scale` |
| Force/velocity token | `f * scale` |
| Masks, material, gravity flag, coefficient, volume | Preserve |

The positive isotropic scale preserves directions and relative phase placement.
Do not center phases independently. This scaling matches the learned model's
input tokens; it is not a complete SI similarity transformation.

For gas-only and interaction-OFF comparisons, fit to the original joint scene
before disabling the solid slot or cross-attention. Preserve the original
point/condition ordering. The gas branch's legacy `force` field is a condition
token used with initial velocity, rather than a continuous SI force.

Restore future coordinates with `prediction / scale + center` before projection.
The predictor saves `prediction` and `restored_prediction` in original scene
coordinates, and `prediction_normalized` in model coordinates. A JSON sidecar
stores preprocessing and inference settings. `--normalization none` selects
the legacy interface.

## Trajectory settings

| Setting | Default |
| --- | --- |
| Model | Joint Stage 2 predictor |
| Seed | 42, CPU `torch.Generator` |
| DDIM | 25 steps, CFG 1.0 |
| Scheduler | 1000 training steps, prediction type `sample`, `clip_sample=False` |
| Precision and loading | bfloat16 CUDA autocast, strict state-dict load |
| Prediction | 24 future states |

Append `--prepare-only` to inspect adapted inputs on CPU without weights.
Use `--phase-mode gas` or `--interaction off` with the same original scene
input and seed for controlled comparisons. A matching PAVG checkpoint and
model configuration are required for prediction.

## Projection and timing

Native-camera inputs include registered world points, source indices, a shared
camera/crop and the initial geometry transform. Regular interpolation places
the initial state and 24 future states at even video-frame indices, and their
midpoints at odd indices. Dense points follow the mean displacement of their
eight nearest retained drivers; frame 0 is checked against registered geometry.

The projector uses native projection when registered fields exist; otherwise
it uses the supplied local calibration and interpolation. Background anchors
retain fixed XY coordinates and z=2.

Tracking controls and videos have **49 frames, 24 fps, 720 x 480**.
First-to-last time is `(49-1)/24 = 2` seconds. MP4 duration is `49/24`, about
2.041667 seconds. Playback fps does not change the predicted physical states.

## DaS settings and checks

Video generation uses **seed 20260910**, **50 steps**, **CFG 6.0**, 49 frames,
bfloat16, the CogVideoX DPM scheduler with trailing timestep spacing and dynamic
CFG. CPU offload is disabled by default.

Seed Python `random`, NumPy, PyTorch and CUDA before loading/encoding tracking
maps: the tracking VAE samples its latent from the global RNG independently
of the video-noise generator. Seeding video noise alone does not reproduce
these settings.

The runner decodes all tracking frames, rejects controls with an unexpected
frame count, and verifies generated frame count/fps. `video --check-only`
checks media without loading weights. Use `--cpu-offload` to keep inactive
modules on CPU when GPU memory is limited.

[scene_prompts.json](../examples/scene_prompts.json) contains example prompts
for different materials and motions. Match the prompt to the input image and
trajectory controls.

Mask-based image initialization approximates depth in 2.5D. Evaluate predicted
trajectories against simulation targets separately from video appearance.
