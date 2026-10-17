# Third-party dependencies

PAVG builds on PhysCtrl, Diffusion as Shader, CogVideoX and Diffusers.
Please retain their notices and follow the applicable source and model terms.

## PhysCtrl

Install
[PhysCtrl at the required version](https://github.com/cwchenwang/physctrl/tree/803264f2901e9f14481ab366991976b517571178)
using the commands in the [README](../README.md#installation).
The default directory is `third_party/physctrl`. Set `PAVG_PHYSCTRL_ROOT`
to use another location.

PAVG loads this dependency through `pavg_external.py`; use the specified
version for compatibility.

## Video generation

Obtain video weights separately from the
[Diffusion as Shader project](https://github.com/IGL-HKUST/DiffusionAsShader)
and follow its installation and model instructions.
DaS uses Apache-2.0 for its source; the other dependencies and model weights
retain their own terms.
