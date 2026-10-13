"""PhysCtrl runtime with verified PAVG extensions; see docs/THIRD_PARTY.md."""
from pathlib import Path as _Path
import sys as _sys
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2]))
from pavg_external import execute_dependency as _execute
_execute('src/utils/physics.py', globals())
