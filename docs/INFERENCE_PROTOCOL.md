# Accepted scene-video inference protocol

`PAVG-normalized32-scaled-conditions-20261011` is the scene-video version
accepted after trajectory/video review. It is separate from the fixed
trajectory benchmark in [Data and Evaluation](DATA_AND_EVALUATION.md).

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
Do not center phases independently. This is the frozen learned-model token
policy, not a complete SI similarity transformation.

For gas-only and interaction-OFF diagnostics, fit to the original joint scene
before disabling the solid slot or cross-attention. Preserve the original
point/condition ordering. The gas branch's legacy `force` field is a condition
token used with initial velocity; it is not a newly defined continuous SI force.

Restore future coordinates with `prediction / scale + center` before projection.
The predictor saves `prediction` and `restored_prediction` in original scene
coordinates, and `prediction_normalized` in model coordinates. Its sidecar
records the transform, input/config/checkpoint hashes, code commit and UTC
times. `--normalization none` explicitly selects the legacy interface.

## Trajectory settings

| Setting | Accepted value |
| --- | --- |
| Model | Stage 2 v4, checkpoint-60000 |
| Checkpoint SHA256 | `654439ea5b94bf699128f018cd2fe70d01cf849e1dbdf6d2c030ce3254210a30` |
| Seed | 42, CPU `torch.Generator` |
| DDIM | 25 steps, CFG 1.0 |
| Scheduler | 1000 training steps, prediction type `sample`, `clip_sample=False` |
| Precision and loading | bfloat16 CUDA autocast, strict state-dict load |
| Prediction | 24 future states |

Use `--expected-checkpoint-sha256` to enforce the reference weight identity.
Append `--prepare-only` to inspect adapted inputs on CPU without weights.
Use `--phase-mode gas` or `--interaction off` with the same original scene
input and seed for diagnostics. Weights remain external; a hash is not a
public download link.

## Projection and timing

Native-camera inputs retain registered world points, source indices, shared
camera/crop and the initial geometry transform. Regular interpolation places
the initial state and 24 future states at even video-frame indices, and their
midpoints at odd indices. Dense points follow the mean displacement of their
eight nearest retained drivers; frame 0 is checked against registered geometry.

`scene08a`, `scene08b` and `scene08c` retain their historical local projection
and interpolation. The route map is explicit in
[inference_protocol.json](../configs/inference_protocol.json). The projector
uses native projection when registered fields exist, otherwise the supplied
local calibration. Background anchors retain fixed XY and z=2.

Tracking controls and videos have **49 frames, 24 fps, 720 x 480**.
First-to-last time is `(49-1)/24 = 2` seconds. MP4 duration is `49/24`, about
2.041667 seconds. Playback fps does not change the predicted physical states.

Four accepted scene inputs (`scene03a`, `scene08a`, `scene08b`, `scene08c`)
were revised before rendering. Their geometry/conditions belong to the input
assets; the runner does not inject scene-name-dependent translations or
material changes.

## DaS settings and checks

The accepted backend uses **seed 20260910**, **50 steps**, **CFG 6.0**, 49
frames, bfloat16, the CogVideoX DPM scheduler with trailing timestep spacing
and dynamic CFG. CPU offload is disabled by default.

Seed Python `random`, NumPy, PyTorch and CUDA before loading/encoding tracking
maps: the tracking VAE samples its latent from the global RNG independently
of the video-noise generator. Seeding video noise alone does not reproduce
this protocol.

The runner decodes/checks all tracking frames, rejects short/long controls,
and verifies generated frame count/fps. `video --check-only` checks media
without loading weights; `--cpu-offload` records a changed execution setting.
The accepted material-aware prompts are in
[scene_prompts.json](../examples/scene_prompts.json). Appearance also depends
on the prompt and initial image, so compare them alongside trajectories.

The example images are input illustrations, not the complete 32-scene assets.
Qualitative acceptance and input adaptation do not certify ground-truth
physics. The fixed benchmark inputs/results remain separate, and no new GPU
synthesis or benchmark is claimed by this source release's CPU verification.
