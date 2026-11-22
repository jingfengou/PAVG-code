# Phase trajectory predictors

Run commands from the repository root. The trajectory and video stages have separate weights.
Install PyTorch and torchvision for your CUDA environment, then `pip install -r requirements.txt`.
Obtain the pinned PhysCtrl dependency and verify its source extensions. See [third-party dependencies](THIRD_PARTY.md).

```bash
git clone --filter=blob:none --no-checkout https://github.com/cwchenwang/physctrl.git third_party/physctrl
git -C third_party/physctrl checkout 803264f2901e9f14481ab366991976b517571178
python scripts/check_dependencies.py
python scripts/check_predictors.py
```

## Models and checkpoints

| YAML in `configs/predictors/` | Predictor | Training and checkpoint used |
| --- | --- | --- |
| `solid_prior.yaml` | Seven-token solid Stage 1 prior, eight blocks, width 256 | 60,000 updates, global batch 32 |
| `pavg.yaml` | Separate solid and gas trunks, spatial and temporal CPI in both directions | 60,000 updates, global batch 32, final checkpoint |
| `without_cpi.yaml` | Same priors and Stage 2 losses, both CPI pathways removed throughout training | 15,000 updates, global batch 128, final checkpoint |
| `physctrl_start.yaml` and `physctrl.yaml` | Shared eight-block PhysCtrl, width 256, four heads | 95,479 updates, validation-selected step 81,250 |
| `unified.yaml` | Shared nine-block capacity comparison, width 384, six heads | 94,229 updates at global batch 64, validation-selected step 90,000 |

The PAVG solid trunk encodes modulus, Poisson ratio, force plus gravity embedding, force application point, gravity flag, material class, and floor height. The gas trunk uses its four-token Stage 1 layout and pointwise initial velocity. Each CPI direction has separate parameters and zero-initialized gates. Single-phase inference runs the active trunk. Shared predictors process each present phase separately with the same weights.

Reference SHA-256 for `model.safetensors`:

| Predictor | SHA-256 |
| --- | --- |
| PAVG | `e77938dbda860453713cb246dbc0729283cbcfc5066b74c908e0f4eabc4ae95f` |
| Without CPI | `95c635c972103be5697165a189841cab8896276e3af12339894b8df26d3eba36` |
| PhysCtrl | `abacc5266438173e33896538be4ff4332f4beb5a656bb8bdbb295c928c664651` |
| Capacity comparison | `ae1d84a3793ded268f1b84a3dea59267abffce10a58d1baeec1650b3498fef16` |

The source distribution contains code and configuration. Supply the matching predictor weights or train them below. Earlier four-token solid checkpoints use `configs/stage2.yaml`. The CPU check accepts `--weights CONFIG CHECKPOINT` to verify a supplied checkpoint strictly.

## Training

Place the canonical HDF5 samples under `datasets/`, manifests under `manifests/`, and the material sidecar at `metadata/composite_object_physics_full.h5`. Combined manifests use `{"path": "...", "phase": "gas"}` records, with phase `solid`, `gas`, or `interaction`. Paths can be relative to `datasets/`. Use a consistent sample path across split lists. Training plan order follows the original manifest order. See [dataset format](DATASET.md).

```bash
python scripts/configure.py --data-root datasets --manifest-root manifests --run-root runs \
  --metadata-sidecar metadata/composite_object_physics_full.h5
python scripts/prepare_predictor_plans.py
python scripts/run.py train-stage1 --num-processes 8 -- --config configs/local/stage1_gas.yaml
python scripts/run.py train-stage2 --num-processes 8 -- --config configs/predictors/solid_prior.yaml
python scripts/run.py train-stage2 --num-processes 8 -- --config configs/predictors/pavg.yaml
python scripts/run.py train-stage2 --num-processes 8 -- --config configs/predictors/without_cpi.yaml
```

`train-stage2` invokes the phase-aware trainer. `solid_prior.yaml` selects its solid-only Stage 1 architecture. PAVG and the variant without CPI initialize from the same two priors. Each consumes 1.92 million Stage 2 examples. The portable implementation uses standard PyTorch attention and gradient checkpointing. Peak memory and speed depend on hardware and kernels.

For a shared predictor, retain 1,920,000 parent exposures from each of Stage 1 solid, Stage 1 gas and Stage 2, including repeated occurrences. The reference catalogs contain 508,100, 99,800 and 707,678 rows. Each interaction occurrence expands to two independent 2,048-point training examples. The resulting 2,461,340 gas and 3,569,294 solid exposures have loss weights 1.2250713 and 0.8447937, respectively. The last batch contains 22 zero-loss fillers. The classifier-free sampler and physics losses respect phase presence. The shared predictors use zero coupling loss.

On a multi-node cluster, launch one Accelerate process group across the **total** 16 GPUs using the appropriate local Accelerate configuration on each node. For a single 16-GPU host:

```bash
python scripts/run.py train-stage2 --num-processes 16 -- --config configs/predictors/physctrl_start.yaml
python scripts/run.py train-stage2 --num-processes 16 -- --config configs/predictors/physctrl.yaml
python scripts/run.py train-stage2 --num-processes 16 -- --config configs/predictors/unified.yaml
```

The PhysCtrl start configuration stops at step 2,500 with global batch 32. The continuation reads that full checkpoint with global batch 64 and an exposure-scaled scheduler. Validation covers all 500 solid, 100 gas and 100 interaction examples with 25 DDIM steps. Checkpoint selection maximizes the overall vIoU, then minimizes CD and Corr on ties. Groups receive weights 0.25, 0.25 and 0.50. Test data do not enter this selection.

## Trajectories and videos

The supplied `examples/interaction/` contains an initial image, masks, phase geometry and controls. Use the conditions NPZ for the recorded force and initial velocity. For inputs already in benchmark coordinates, use `--normalization none`. For an external scene requiring a shared bounding-box transform, use `--normalization joint`. Both return trajectories in the input coordinate system.

```bash
python scripts/run.py predict -- --config configs/predictors/pavg.yaml \
  --checkpoint runs/pavg/checkpoint-60000/model.safetensors \
  --phase-inputs examples/interaction/phase_inputs.npz \
  --condition-input examples/interaction/condition_input.npz --normalization none \
  --attention-backend math --seed 42 --num-inference-steps 25 --output outputs/example/prediction.npz

python scripts/run.py project -- --phase-inputs examples/interaction/phase_inputs.npz \
  --solid-mask examples/interaction/solid_mask.png --gas-mask examples/interaction/gas_mask.png \
  --prediction outputs/example/prediction.npz --output-dir outputs/example/tracks
python scripts/run.py render-tracks -- --tracks outputs/example/tracks/tracks.npy \
  --output-dir outputs/example/tracking --write-mp4
python scripts/run.py video -- --input-image examples/interaction/input.png \
  --tracking-video outputs/example/tracking/tracks_tracking.mp4 --checkpoint weights/das \
  --prompt "A ball moves through dust." --seed 20260910 --output-dir outputs/example/video
```

To run PhysCtrl on the same example, replace the YAML with `configs/predictors/physctrl.yaml` and checkpoint with `runs/physctrl/best/model.safetensors`. The runner resets the CPU noise generator to seed 42 for each present phase, denoises each independently, then returns both predictions to the renderer. PAVG samples paired trajectories together. The projection and video commands apply to either output. DaS uses 49 frames at 720 by 480, 24 fps, 50 diffusion steps, and guidance 6.0. Its video checkpoint is obtained separately.

## Trajectory evaluation

```bash
python scripts/run.py eval-stage2 -- --config configs/predictors/pavg.yaml \
  --checkpoint runs/pavg/checkpoint-60000/model.safetensors --test-manifest manifests/combined_test_paths.json \
  --end-index 700 --num-inference-steps 25 --seed 20260804 --output results/pavg.json
python scripts/run.py eval-split -- shard --config configs/predictors/physctrl.yaml \
  --checkpoint runs/physctrl/best/model.safetensors \
  --checkpoint-sha abacc5266438173e33896538be4ff4332f4beb5a656bb8bdbb295c928c664651 --output results/physctrl
python scripts/run.py eval-split -- aggregate --output results/physctrl
```

For distributed independent evaluation, pass distinct `--rank` values and the same `--world-size` to each shard and use the same world size for aggregation. Test seed is `20260804 + manifest index`. Shared predictors concatenate the independently generated point sets only when computing interaction metrics.

## Liquid extension

The qualitative liquid experiment uses a separately trained eight-block, width-256, four-head predictor with 2,048 points and 24 future states. The four materials are water, milk, edible oil, and honey. The gravity-only split contains 11,167 training trajectories, 635 validation trajectories and 663 held-out test trajectories. The model trains for 60,000 updates, with density and dynamic viscosity as material inputs, floor-relative positions and displacement scales fitted on the training set. The objective combines axis-balanced position and interval-velocity losses, a floor penalty and ground-truth contact height weighted by 10. The four illustrated examples come from validation. XY shows the fall, with Y measuring height, and XZ shows horizontal spreading. Liquid-specific training code, data and weights are outside this source distribution.
