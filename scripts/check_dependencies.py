"""Verify an independently acquired PhysCtrl checkout and all PAVG extensions."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pavg_external import compose_source, dependency_root, extension_manifest


def main():
    verified = []
    for relative in extension_manifest()["files"]:
        source = compose_source(relative)
        compile(source, relative, "exec")
        verified.append(relative)
    print(json.dumps({"status": "passed", "dependency_root": str(dependency_root()),
                      "verified_extension_modules": verified}, indent=2))


if __name__ == "__main__":
    main()
