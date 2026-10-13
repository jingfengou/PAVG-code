# October 13, 2026: accepted inference and public source snapshot

- Added shared normalization, scaled conditions and restored predictions.
- Added native camera projection and retained the three legacy local routes.
- Fixed trajectory defaults to seed 42 / DDIM 25 / CFG 1.0; DaS defaults to
  seed 20260910 / 50 steps / CFG 6.0 with global RNGs seeded before VAE encoding.
- Enforced 49 frames at 24 fps and added CPU preparation/media checks.
- Kept unchanged PhysCtrl code external, with verified PAVG source extensions.
- Kept fixed benchmark procedures separate; weights and datasets remain external.

# Public source update: 2026-09-25

## Motivation

The portable export lagged behind the research implementation's corrected
training and fixed-evaluation protocols. This update repairs those differences
before replacing the previous main source tree. Datasets, checkpoints,
private operations scripts and experimental outputs remain outside Git.
The implementation and repository are named PAVG, matching the manuscript
method: PAVG.

## Corrections

- Resolve Stage 1 source snapshots from the training file, not the launch
  directory, and retain the local utilities needed by the snapshot.
- Average complete active Stage 2 branch objectives before adding coupling,
  including the corrected training-contract version and advection setting.
- Export the exposure-matched Unified sampler and its final per-phase audit:
  24 processes, 2 solid + 2 gas per rank, 40K successful updates, 1.92M
  examples per phase. Gas never receives solid floor/deformation losses.
  Explicit research recovery is supported with auditable cursors and labels;
  an ordinary resume without a cursor is rejected.
- Restore Stage 1 manifest-order checks, per-sample CPU seeds, prediction
  preservation, hashes and shard ranges; fix non-full tail-batch indices.
- Report fixed test700 Overall and add a strict sample-level shard merger.
  Reject incomplete coverage, incompatible provenance, wrong phase counts,
  invalid metrics and changed sample identities.
- Remove the unused `torch_cluster.fps` import. No point-sampling algorithm
  was replaced; the released loader already uses NumPy sampling.

## Verification

Run from the repository root in an installed training environment:

```bash
python scripts/audit_release.py
git diff --check
```

The public source release includes the portable release audit but does not
ship the internal regression and runtime smoke test suite. The audit checks
source syntax, private machine paths, generated artifacts, secrets and
documentation links. It is not a quality score or a replacement for a real
checkpoint benchmark.

Local validation used Python 3.10.19, PyTorch 2.11.0+cu128, torchvision 0.26.0,
Diffusers 0.33.0, Accelerate 1.1.1, NumPy 2.2.6 and h5py 3.16.0. This reused
environment is not the exact pinned reference environment. No full multi-GPU
training, checkpoint test700 rerun or DaS video synthesis was performed for
this source update; those remain runtime/release verification steps on the
intended Linux/CUDA setup.

## Distribution Scope

This update does not publish weights, dataset access credentials, or new
benchmark values. Existing third-party provenance and notices are retained.
It does not add a blanket project license or resolve the upstream permission
check described in [third-party notes](THIRD_PARTY.md). Upstream code and
weight redistribution terms still require separate confirmation.
