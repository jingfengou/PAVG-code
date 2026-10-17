# Data and evaluation contract

## Local layout

`scripts/configure.py` expects the following manifest basenames:

```text
solid_train_paths.json       solid_val_paths.json       solid_test_paths.json
gas_train_paths.json         gas_val_paths.json         gas_test_paths.json
unified_train_paths.json     unified_val_paths.json     unified_test_paths.json
combined_train_paths.json    combined_val_paths.json    combined_test_paths.json
```

Each manifest is an ordered JSON array of sample paths. Stage 1 unified
contains solid and gas samples, not interaction samples. Stage 2 combined
contains all three settings. Dataset-root-relative paths are portable;
machine-specific manifests belong outside Git. Do not reshuffle test manifests,
change sample identity, or regenerate splits for each model.

The exposure-matched Unified trainer reads `solid_train_paths.json` and
`gas_train_paths.json` separately and interleaves them deterministically.
It does not consume `unified_train_paths.json`; the unified validation/test
manifests still contain both phases. The generated training recipe uses 24
processes, local batch 2 solid + 2 gas, and 40K successful updates. Never
change the phase mixture or world size without rechecking the exposure budget.
Solid and gas loader overrides retain their respective gravity handling.

Unified checkpoints must not resume by silently restarting the sampler.
An explicitly audited `phase_balanced.recovery` configuration can supply
`policy: replay_skips_replace_invalid_labels`, `consumed_batches`,
`skipped_batches`, `completed_updates`, `replacement_seed`, and `rng_seed`.
The cursor is measured in local batches; skipped entries must be sorted,
unique, and precede it. The checkpoint step must match `completed_updates`.
Only invalid target labels may be replaced, by deterministic draws from the
same phase and locked split; invalid input or I/O remains fatal. Recovery
writes per-rank sample/replacement logs and records its policy in the final
exposure audit. Do not invent a cursor from the checkpoint filename alone.

## Raw HDF5 and loader output

The Stage 1 loader consumes PhysCtrl-compatible `x`, `v`, `F`, `C`, `vol`,
`E`, `nu`, `mat_type`, `drag_force`, `drag_point`, `drag_mask`,
`floor_height` and `gravity` fields, with the compatibility/default handling
defined in `src/dataset/traj_dataset.py`. Not every optional field is mandatory
for every material. Gas is material ID 4; rigid solids use ID 3. Gas has
compatibility `F`/`C` fields but receives neither the solid MPM deformation loss
nor the solid floor loss, including in mixed training and validation batches.

`points_src`, `points_tgt`, `init_vel`, masks and physical-condition tensors
are **loader outputs**, not names that must replace the raw HDF5 fields.
Positions are normalized using the configured `norm_fac`; initial velocities,
force coefficients, volume and gravity have material-specific handling.
Do not pre-normalize raw files a second time.

The joint loader `interaction/dataset/object_gas_dataset.py` handles the
canonical composite protocol and single-phase samples. Its phase-slot order
is `[gas, solid]`. Absent phases are zero-filled and identified by
`phase_presence`. The reader also derives interval-aligned gas displacement,
alive masks and interaction labels from stored simulation fields.

Composite solid conditions must be recoverable from embedded metadata, the
original solid sample, or the compatible metadata sidecar. Supply the sidecar
to `scripts/configure.py` when using the formal composite dataset. The public
loader deliberately does not rewrite an old server prefix to a different
hard-coded server prefix. If relocating source HDF5 files, preserve relative
source paths or use the matching sidecar and verify one sample per setting.

## Input-only examples

Image inference accepts numeric NPZ arrays: `points_src` has shape `[2,N,3]`,
`phase_presence` has shape `[2]`, and condition arrays follow the same point
order. `initialize_phase_points.py` writes projection metadata so that local
points can be mapped back to image pixels. Stored force/velocity controls are
used by `infer_phase_scene.py` when provided. CLI defaults are demonstrations,
not ground-truth physical annotations.

Do not load pickle-containing NPZ/NPY artifacts from untrusted sources. The
small published NPZ inputs contain numeric arrays only. The generated track
payload uses the legacy trusted-local NumPy dictionary format for DaS.

## Fixed evaluation

The formal combined test has 500 solid, 100 gas and 100 interaction samples.
Overall metrics weight these groups by `0.25 / 0.25 / 0.50`; they are not the
unweighted mean over all 700 samples. Interaction's joint metrics and its
separate gas/solid metrics are different quantities and should be retained.

The fixed Stage 2 evaluator uses DDIM25, per-sample seeds based on the global
manifest index, and the shared definitions in `src/utils/physctrl_metrics.py`.
Higher vIoU is better; lower CD and Corr are better. CD sums the two directed
squared nearest-neighbor means. Corr measures matched-point Euclidean error.

The index range is half-open. Use the full manifest and disjoint
`--start-index/--end-index` ranges for parallel evaluation. If using a sliced
manifest, provide the original `--seed-index-offset`. Preserve per-sample
records and check full, nonduplicated coverage before aggregating. The main
and `--disable-cross-attention` evaluations must use the same checkpoint,
inputs, seeds and metric protocol.

Stage 2 JSON includes group counts, `complete_fixed_test`, the separate
interaction streams, and `metrics.overall`. Overall is null until all 700
indices and the 500/100/100 phase composition are present.
`scripts/run.py merge-stage2` requires shards evaluated against the same
full manifest with zero seed-index offset. Sliced-manifest outputs remain
usable for diagnostics but are rejected by the formal merger. The merger
checks checkpoint/config/evaluator/metric/aggregation hashes, DDIM25, seed
20260804, the cross-attention toggle, each source path and per-sample seed,
finite metrics, and exact nonduplicated coverage. Group assignment uses
`phase_presence`, never filename heuristics. All averages are recomputed
from individual samples; JSON output is accompanied by a SHA256 sidecar.

Stage 1 defaults to per-sample CPU seeds `20260820 + original manifest index`.
This mode requires evaluation batch size one and supports half-open ranges
of the full manifest. The evaluator enforces manifest membership/order,
records source/config/checkpoint hashes and input/target hashes, and accepts
`--save-predictions` for independent recomputation. Historical stream mode
starts at index zero, uses the configured seed unless explicitly overridden,
and must not be presented as the current per-sample benchmark.

## Reproducible evaluation

Save each evaluated checkpoint with its config, exact train/test manifests,
code and dependency versions, seeds, and per-sample output JSON. Keep these
inputs fixed when comparing models or ablations, and recompute aggregate
metrics from complete per-sample results.
