<h1 align="center">PAVG: Phase-Aware Video Generation<br>for Physics-Grounded Dynamics and Interactions</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2610.11791"><img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&amp;logoColor=white" alt="arXiv paper" height="22"></a>
  <a href="https://jingfengou.github.io/PAVG/"><img src="https://img.shields.io/badge/Project_Page-Website-2f855a?logo=googlechrome&amp;logoColor=white" alt="Project page and videos" height="22"></a>
  <a href="https://huggingface.co/irony2025/PAVG"><img src="https://img.shields.io/badge/Hugging_Face-Models-007ec6?logo=huggingface&amp;logoColor=FFD21E" alt="Hugging Face pretrained models" height="22"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-6f42c1?logo=apache&amp;logoColor=white" alt="License: Apache 2.0" height="22"></a>
</p>

<p align="center">Controllable video generation with predicted 3D trajectories for solids, gas and solid–gas interactions.</p>

<a name="installation"></a>

## 📦 Installation

Use Python 3.10 on Linux with an NVIDIA GPU. Install PyTorch 2.5.1 and
torchvision 0.20.1 for your CUDA version, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Install the PhysCtrl dependency:

```bash
git clone --config core.autocrlf=false --filter=blob:none --no-checkout \
  https://github.com/cwchenwang/physctrl.git third_party/physctrl
git -C third_party/physctrl checkout 803264f2901e9f14481ab366991976b517571178
python scripts/check_dependencies.py
```

To use an existing checkout of this version, set `PAVG_PHYSCTRL_ROOT` to its
directory.

<a name="pretrained-models"></a>

## 🤖 Pretrained Models

Download the [pretrained PAVG models](https://huggingface.co/irony2025/PAVG) into `weights/pavg/`:

```bash
hf download irony2025/PAVG --local-dir weights/pavg
```

| Model file | Use |
| --- | --- |
| `model.safetensors` | Joint solid, gas and interaction trajectory prediction |
| `stage1/solid/model.safetensors` | Solid trajectory prediction and Stage 2 training initialization |
| `stage1/gas/model.safetensors` | Gas trajectory prediction and Stage 2 training initialization |

Start with the joint model and `weights/pavg/configs/stage2.yaml` to run the
example below. The Stage 1 downloads are optional for joint inference.
Obtain the DaS video-generation weights separately from their authors.

<a name="dataset"></a>

## 📂 Dataset

Prepare HDF5 samples and JSON path lists for solid, gas and solid-gas data.
See [Dataset setup](docs/DATASET.md) for file layout and formats.
Public dataset downloads are not available yet.

The `examples/` directory contains image, mask and initial-condition inputs
for a gas scene and a solid-gas scene.

<a name="image-to-video-generation"></a>

## 🎬 Image to Video Generation

Run the following commands from the repository root:

```bash
# Predict 3D trajectories.
python scripts/run.py predict -- \
  --config weights/pavg/configs/stage2.yaml \
  --checkpoint weights/pavg/model.safetensors \
  --phase-inputs examples/interaction/phase_inputs.npz \
  --condition-input examples/interaction/condition_input.npz \
  --output outputs/example/prediction.npz

# Convert trajectories into video controls.
python scripts/run.py project -- \
  --phase-inputs examples/interaction/phase_inputs.npz \
  --solid-mask examples/interaction/solid_mask.png \
  --gas-mask examples/interaction/gas_mask.png \
  --prediction outputs/example/prediction.npz \
  --output-dir outputs/example/tracks
python scripts/run.py render-tracks -- \
  --tracks outputs/example/tracks/tracks.npy \
  --output-dir outputs/example/tracking --write-mp4

# Generate the video.
python scripts/run.py video -- \
  --input-image examples/interaction/input.png \
  --tracking-video outputs/example/tracking/tracks_tracking.mp4 \
  --checkpoint weights/das \
  --prompt "A ball moves through a cloud of dust, displacing the dust." \
  --output-dir outputs/example/video
```

The generated video is saved to `outputs/example/video/result.mp4`.
For custom images and motion controls, see [Inference](docs/INFERENCE.md).
To try input preparation and tracking rendering without weights, see
[Check your installation](docs/VERIFICATION.md).

<a name="training-and-evaluation"></a>

## 🚀 Training and Evaluation

### Prepare configs

Set your dataset, manifest and output directories:

```bash
python scripts/configure.py \
  --data-root datasets --manifest-root manifests --run-root runs \
  --metadata-sidecar metadata/composite_object_physics_full.h5
```

### Train

Train the solid and gas models, then the joint model:

```bash
python scripts/run.py train-stage1 --num-processes 8 -- \
  --config configs/local/stage1_solid.yaml
python scripts/run.py train-stage1 --num-processes 8 -- \
  --config configs/local/stage1_gas.yaml
python scripts/run.py train-stage2 --num-processes 16 -- \
  --config configs/local/stage2.yaml
```

Set `initialization.solid_checkpoint` and `initialization.gas_checkpoint` in
`configs/local/stage2.yaml` to the Stage 1 checkpoints you want to use.

### Evaluate

```bash
python scripts/run.py eval-stage1 -- \
  --config configs/local/stage1_gas.yaml \
  --checkpoint runs/stage1_gas/checkpoint-60000/model.safetensors \
  --test-manifest manifests/gas_test_paths.json \
  --dataset-label gas --save-predictions --output results/gas.json

python scripts/run.py eval-stage2 -- \
  --config configs/local/stage2.yaml \
  --checkpoint runs/stage2/checkpoint-60000/model.safetensors \
  --test-manifest manifests/combined_test_paths.json \
  --end-index 700 \
  --save-predictions --output results/joint.json
```

Set `--end-index` to the number of samples to evaluate. Results are written
to `--output`; `--save-predictions` also saves the predicted trajectories.

<a name="citation-and-acknowledgments"></a>

## 📚 Citation and Acknowledgments

The manuscript is [Phase-Aware Video Generation for Physics-Grounded Dynamics
and Interactions](https://arxiv.org/abs/2610.11791).

This implementation builds on PhysCtrl, Diffusion as Shader, CogVideoX and
Diffusers. See [third-party dependencies](docs/THIRD_PARTY.md) for attribution.

<a name="license"></a>

## 📄 License

PAVG's original source code is licensed under the
[Apache License 2.0](LICENSE). The PAVG trajectory checkpoints on
[Hugging Face](https://huggingface.co/irony2025/PAVG) use the same license.
Third-party code and model weights retain their respective terms; see
[third-party dependencies](docs/THIRD_PARTY.md).
