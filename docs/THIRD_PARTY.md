# Third-party dependencies

The PhysCtrl development reference is
[`803264f2901e9f14481ab366991976b517571178`](https://github.com/cwchenwang/physctrl/tree/803264f2901e9f14481ab366991976b517571178).
The upstream reference has no root code license as inspected on October 13,
2026. Unchanged upstream source is external to this release. Acquire it from
its authors and follow its applicable source and model terms.

Overlapping files in `src/` and `libs/das/` use small entrypoints.
`pavg_external.py` checks the external source hash, applies PAVG changes in
memory and checks the result. `dependencies/physctrl_extensions.json` contains
PAVG insertion/replacement text, reference offsets and hashes; unchanged
implementation and deletion context are absent. The loader never rewrites or
automatically downloads dependency files.

Acquire the pinned dependency as shown in the [README](../README.md).
The default location is `third_party/physctrl`; an existing matching checkout
can be selected with `PAVG_PHYSCTRL_ROOT`. All required hashes are checked,
including source archives. A different Git commit or file hash stops loading.
Original class names and checkpoint parameter names are preserved.

PAVG changes include physical adapters, phase-specific branches, cross-phase
attention, gas/interaction objectives, canonical data handling, evaluation,
shared normalization and the trajectory-to-video interface.

DaS, CogVideoX and Diffusers retain their source/model terms and notices.
The [original DaS project](https://github.com/IGL-HKUST/DiffusionAsShader) uses
Apache-2.0; this is not asserted as a blanket license for PhysCtrl or weights.
No blanket MIT license is asserted for the composed runtime. Model weights
and datasets are not bundled.

Input examples are project-generated images and initial conditions.
Simulation targets and solver builds are not bundled.
