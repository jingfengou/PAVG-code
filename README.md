# PAVG

**Phase-Aware physics-grounded controllable Video Generator**

PAVG predicts physical 3D point trajectories for single-phase motion and
solid-gas interaction, then projects the trajectories to control a pretrained
video generator. Solid and gas use separate diffusion branches; Stage 2
initializes these branches from their corresponding Stage 1 models.

This source release contains trajectory training, inference, fixed-protocol
evaluation, and the mask-initialized trajectory-to-DaS video interface.
It does not contain datasets, model weights, private launch scripts, or
unverified experimental results. This is a source-only release: it does not
advertise an included checkpoint or certify a completed benchmark run.

[Paper](https://arxiv.org/abs/2610.11791) ·
[Project and videos](https://jingfengou.github.io/PAVG/) ·
[Accepted inference protocol](docs/INFERENCE_PROTOCOL.md)

The scene-video pipeline includes shared normalization and condition scaling,
restores predictions before projection, and exports 49 frames at 24 fps
(2 seconds between the first and last frames).

## Installation

Use Linux with an NVIDIA GPU for training and video generation. Reuse a
compatible PhysCtrl environment, or install PyTorch and torchvision for your
CUDA version before the dependencies below. Python 3.10 is the reference
training environment, with PyTorch 2.5.1 and torchvision 0.20.1 built for the
same CUDA version. The CPU input/track examples
do not require a GPU.

```bash
python -m pip install -r requirements.txt
```

Acquire the pinned PhysCtrl reference separately. Unchanged upstream source
is external; this repository supplies verified PAVG source extensions.

```bash
git clone --config core.autocrlf=false --filter=blob:none --no-checkout \
  https://github.com/cwchenwang/physctrl.git third_party/physctrl
git -C third_party/physctrl checkout 803264f2901e9f14481ab366991976b517571178
python scripts/check_dependencies.py
```

For an existing matching checkout, set `PAVG_PHYSCTRL_ROOT` to its location.
See [third-party provenance](docs/THIRD_PARTY.md).

The released loader uses NumPy point sampling and does not require
`torch-cluster`. Optional data-generation and baseline dependencies belong to the research
workspace, not this minimal runtime. See [third-party notes](docs/THIRD_PARTY.md).

## Pretrained Models

PAVG requires a matching Stage 2 checkpoint and config for joint trajectory
inference. Training Stage 2 additionally requires specialized solid and gas
Stage 1 checkpoints. Video generation uses DaS weights in the format expected
by its CogVideoX tracking pipeline.

Weights are external to Git. Put them in `weights/` or pass their locations
explicitly. Public weight and dataset downloads are not included in this
source release. The accepted checkpoint hash is recorded in
[the protocol](configs/inference_protocol.json); it identifies an artifact,
not a download link. Historical checkpoints must not be relabeled as corrected runs.

## Dataset

The trajectory interface supports solid-only, gas-only, and solid-gas HDF5
samples. Fixed JSON manifests determine the split and sample order. Supply
paths relative to your dataset root, or absolute paths in your local manifests.
The input schema, manifest naming, normalization and metadata requirements are
described in [Data and Evaluation](docs/DATA_AND_EVALUATION.md).

The repository includes two small input examples: a gas-only scene and a
solid-gas scene. They contain clean images, masks and numeric initial
conditions, **not ground-truth simulations or final-model result claims**.
See [examples](examples/README.md).

## Image to Video Generation

Run all commands from this directory. All file locations are command-line
arguments or generated local configuration; no server layout is assumed.

```bash
# Sample phase-specific points from masks (gas-only: omit --solid-mask).
python scripts/run.py initialize -- \
  --solid-mask examples/interaction/solid_mask.png \
  --gas-mask examples/interaction/gas_mask.png \
  --output-dir outputs/example/init

# Predict with your matching checkpoint. The provided example preserves its
# recorded point ordering and conditions.
python scripts/run.py predict -- \
  --config configs/stage2.yaml \
  --checkpoint weights/pavg/model.safetensors \
  --phase-inputs examples/interaction/phase_inputs.npz \
  --condition-input examples/interaction/condition_input.npz \
  --output outputs/example/prediction.npz --normalization joint --seed 42

# Project, merge and render the trajectories.
python scripts/run.py project -- \
  --phase-inputs examples/interaction/phase_inputs.npz \
  --solid-mask examples/interaction/solid_mask.png \
  --gas-mask examples/interaction/gas_mask.png \
  --prediction outputs/example/prediction.npz \
  --output-dir outputs/example/tracks
python scripts/run.py render-tracks -- \
  --tracks outputs/example/tracks/tracks.npy \
  --output-dir outputs/example/tracking --write-mp4

# Synthesize the video; replace the prompt to match your input and control.
python scripts/run.py video -- \
  --input-image examples/interaction/input.png \
  --tracking-video outputs/example/tracking/tracks_tracking.mp4 \
  --checkpoint weights/das \
  --prompt "A ball moves through a cloud of dust, displacing the dust." \
  --output-dir outputs/example/video --seed 20260910 --fps 24
```

The first command demonstrates initialization for a new image. The following
commands use the stored example initialization and its matching conditions;
do not apply stored per-point conditions to a differently sampled cloud.
Mask initialization is a 2.5D approximation, not a complete image-to-3D
reconstruction system. Video frame rate and physical simulation time are
separate quantities; do not interpret changing the MP4 frame rate as changing
the trajectory's physical time step.

Use `predict --prepare-only` to inspect adapted inputs on CPU without weights,
and `video --check-only` to validate tracking media without synthesis.
`--phase-mode gas` and `--interaction off` reuse the original joint transform
for controlled diagnostics. See [Inference Protocol](docs/INFERENCE_PROTOCOL.md)
for restoration, interpolation, RNG handling and validation limits.

## Training and Evaluation

Generate machine-local configs rather than editing tracked templates:

```bash
python scripts/configure.py \
  --data-root datasets --manifest-root manifests --run-root runs \
  --metadata-sidecar metadata/composite_object_physics_full.h5

# Each command below is a separate run.
python scripts/run.py train-stage1 --num-processes 8 -- \
  --config configs/local/stage1_solid.yaml
python scripts/run.py train-stage1 --num-processes 8 -- \
  --config configs/local/stage1_gas.yaml
python scripts/run.py train-stage2 --num-processes 16 -- \
  --config configs/local/stage2.yaml
```

The Stage 1 specialist settings use global batch 32 on eight GPUs; Stage 2 uses global
batch 32 on sixteen GPUs. Each stage uses 60,000 optimizer updates.
For multiple machines, pass your local Accelerate configuration with
`--accelerate-config`; rank and rendezvous settings must be provided for your
cluster. Changing the GPU count without adjusting accumulation changes the
effective batch.

Unified Stage 1 is an optional exposure-matched ablation, not the main Stage 2
initialization. Its generated config uses 24 GPUs, two solid and two gas
samples per rank, no gradient accumulation, and 40,000 updates. Each phase
therefore contributes 1,920,000 examples, matching its 60K specialist.
The sampler checks the world size and exact phase counts, and a completed run
writes `phase_balanced_final.json`. Launch it with your cluster's 24-process
Accelerate configuration:

```bash
python scripts/run.py train-stage1 --accelerate-config configs/local/accelerate_24gpu.yaml -- \
  --config configs/local/stage1_unified.yaml
```

Create that Accelerate configuration locally; it is not generated by
`scripts/configure.py`. Non-finite losses or gradients abort the exposure-matched
run instead of silently skipping samples. Resuming Unified requires an explicit
audited recovery cursor; ordinary `resume_from_checkpoint` alone is rejected.

The solid objective uses deformation and contact constraints. The gas
objective uses the divergence proxy, with advection loss disabled.
Stage 2 averages the complete objectives of the active branches, including
their physical terms, then adds the interaction loss. Gas receives neither
the solid deformation term nor the floor term.

```bash
python scripts/run.py eval-stage1 -- \
  --config configs/local/stage1_gas.yaml \
  --checkpoint runs/stage1_gas/checkpoint-60000/model.safetensors \
  --test-manifest manifests/gas_test_paths.json \
  --seed 20260820 --seed-mode per-sample --save-predictions \
  --dataset-label gas --output results/gas.json

python scripts/run.py eval-stage2 -- \
  --config configs/local/stage2.yaml \
  --checkpoint runs/stage2/checkpoint-60000/model.safetensors \
  --test-manifest manifests/combined_test_paths.json \
  --start-index 0 --end-index 700 --batch-size 4 \
  --save-predictions --output results/test700.json
```

Stage 2 reports vIoU, CD and correspondence error (Corr), including separate
gas and solid metrics within interaction samples. Corr is an error, not a
correlation coefficient. Shards must retain the original manifest indices and
use `seed = 20260804 + index`. Complete test700 results include
`0.25 * Solid + 0.25 * Gas + 0.50 * Interaction` as `metrics.overall`.
Partial shards have `metrics.overall = null`.

For parallel evaluation, use the same full manifest and nonoverlapping index
ranges. Merge all resulting JSON files with repeated `--shard` arguments:

```bash
python scripts/run.py merge-stage2 -- \
  --test-manifest manifests/combined_test_paths.json \
  --shard results/shard_0.json --shard results/shard_1.json \
  --output results/test700_merged.json
```

The merger recomputes from individual samples and rejects missing or duplicate
indices, incorrect phase counts, source/seed mismatches, non-finite metrics,
or incompatible checkpoint/config/protocol hashes. Do not average shard means.

Stage 1 defaults to `seed = 20260820 + original manifest index` on a CPU
generator, with `eval_batch_size: 1`. `--start-index` and `--end-index` preserve
that identity across shards. Historical `--seed-mode stream` remains available
for a run starting at index zero; it does not reproduce the per-sample protocol.

## Repository Structure

```text
configs/       Portable Stage 1 and Stage 2 templates
src/           Single-phase diffusion models, loader and physics losses
interaction/   Joint model, loader, losses and inference
libs/das/      Verified external video-backend entrypoints
dependencies/ PAVG source extensions for the pinned external reference
experiments/   Selected fixed evaluators and image/track interface
scripts/       Configuration, launch, input example and release audit
examples/      Small input-only scenes
docs/          Data contract and attribution
```

Run `python scripts/audit_release.py` before packaging to check source files,
private paths, generated artifacts and documentation links. The audit is not a
substitute for the fixed benchmark or a full video synthesis run.
See [release notes](docs/RELEASE_NOTES.md) for the fixes and validation scope.

## Citation and Acknowledgments

The manuscript is [PAVG: Phase-Aware physics-grounded controllable Video
Generator](https://arxiv.org/abs/2610.11791). This snapshot does not claim
conference acceptance.

This implementation builds on PhysCtrl, Diffusion as Shader, CogVideoX and
Diffusers. Retain their notices and follow the source and weight licenses.
See [third-party notes](docs/THIRD_PARTY.md) for dependency boundaries and
pinned-source provenance.
