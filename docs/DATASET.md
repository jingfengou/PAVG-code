# Dataset setup

Store HDF5 samples under a dataset directory and list their paths in JSON
files under a manifest directory:

```text
datasets/
  solid/...
  gas/...
  interaction/...
manifests/
  solid_train_paths.json
  solid_val_paths.json
  solid_test_paths.json
  gas_train_paths.json
  gas_val_paths.json
  gas_test_paths.json
  combined_train_paths.json
  combined_val_paths.json
  combined_test_paths.json
```

Each manifest is a JSON array of paths relative to `datasets/`, or absolute
paths. For example:

```json
["solid/sample_001.h5", "solid/sample_002.h5"]
```

Use the solid and gas lists for single-phase training, and the combined lists
for joint training. Combined lists can contain all three types of sample.

## HDF5 formats

| Samples | Loader |
| --- | --- |
| Solid or gas | [traj_dataset.py](../src/dataset/traj_dataset.py) |
| Solid-gas interaction | [object_gas_dataset.py](../interaction/dataset/object_gas_dataset.py) |

The single-phase loader reads PhysCtrl-style fields such as `x`, `v`, `F`,
`C`, `vol`, `E`, `nu`, `mat_type`, `drag_force`, `drag_point`, `drag_mask`,
`floor_height` and `gravity`. Use the linked loaders when adapting your files.

For interaction samples whose solid material properties are stored separately,
pass the metadata file when generating configs:

```bash
python scripts/configure.py \
  --data-root datasets --manifest-root manifests --run-root runs \
  --metadata-sidecar metadata/composite_object_physics_full.h5
```

The generated configs are saved in `configs/local/`. Adjust dataset and
checkpoint paths there before running the training or evaluation commands
in the [README](../README.md#training-and-evaluation).
