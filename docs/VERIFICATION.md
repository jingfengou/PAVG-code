# Source-release verification: October 13, 2026

The release was checked against retained accepted inputs and outputs on CPU:

- Shared normalization reproduced 64 stored input variants across 32 scenes
  element by element. Restoration reproduced all 64 saved trajectories.
- The public CLI reproduced all 32 main inputs and 32 diagnostic inputs
  (16 gas-only and 16 interaction-OFF), preserving the original joint transform.
- Recomputed projected controls for `giraffeX`, `scene04b`, `scene03a`,
  `scene05a`, `scene09a` and `scene08a` were exactly equal to the retained
  tracks, including phase labels and source indices.
- A small joint model constructed, loaded its state dict strictly, and ran
  two diffusion steps on CPU. This is a runtime smoke check, not an evaluation
  of the full pretrained checkpoint.
- Tracking rendering and media checks produced/validated 49 frames at 24 fps.
  A 48-frame control and an unexpected dependency source hash were rejected.
- All 12 external source modules composed to their recorded source hashes;
  the distributable source audit checked syntax, private paths, credential
  patterns, artifacts, file size and documentation links.

Numeric/runtime checks used Python 3.10.19 and PyTorch 2.11.0 on CPU.
Reference training used PyTorch 2.5.1. The retained-archive array comparison
also passed under NumPy 2.4.1. These checks do not establish GPU bitwise
equivalence across dependency/hardware versions.

No new full-checkpoint GPU synthesis, training or benchmark was run for this
publication. The source release provides no dataset or checkpoint downloads
and does not certify ground-truth physics accuracy.
