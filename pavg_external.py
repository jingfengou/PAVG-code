"""Load a separately acquired, pinned PhysCtrl dependency plus PAVG changes.

Only additions/replacements written for PAVG are stored in this repository.
Unchanged upstream implementation is read from the user's dependency checkout.
No dependency files are rewritten, downloaded, or installed by this loader.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
from functools import lru_cache

ROOT = Path(__file__).resolve().parent
REFERENCE_COMMIT = "803264f2901e9f14481ab366991976b517571178"


@lru_cache(maxsize=1)
def dependency_root() -> Path:
    root = Path(os.environ.get("PAVG_PHYSCTRL_ROOT", ROOT / "third_party" / "physctrl")).resolve()
    if not (root / "src" / "model" / "spacetime.py").is_file():
        raise FileNotFoundError(
            "A separately acquired PhysCtrl checkout is required. "
            "Follow docs/THIRD_PARTY.md and set PAVG_PHYSCTRL_ROOT if needed."
        )
    if (root / ".git").exists():
        commit = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip()
        if commit != REFERENCE_COMMIT:
            raise ValueError(f"Expected PhysCtrl {REFERENCE_COMMIT}, received {commit}")
    return root


def source_digest(text: str) -> str:
    # Normalize checkout line endings; core.autocrlf differs across platforms.
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def extension_manifest() -> dict:
    return json.loads((ROOT / "dependencies" / "physctrl_extensions.json").read_text(encoding="utf-8"))


def compose_source(relative: str) -> str:
    manifest = extension_manifest()
    if manifest["upstream_commit"] != REFERENCE_COMMIT or relative not in manifest["files"]:
        raise ValueError("Unrecognized dependency source or reference commit")
    entry = manifest["files"][relative]
    path = dependency_root() / relative
    text = path.read_text(encoding="utf-8-sig")
    if source_digest(text) != entry["base_sha256"]:
        raise ValueError(f"PhysCtrl dependency source differs from the pinned reference: {relative}")
    lines = text.splitlines(keepends=True)
    cursor, result = 0, []
    for edit in entry["edits"]:
        start, end = edit["start"], edit["end"]
        if not 0 <= cursor <= start <= end <= len(lines):
            raise ValueError(f"Invalid PAVG extension offsets: {relative}")
        result.extend(lines[cursor:start])
        result.append(edit["replacement"])
        cursor = end
    result.extend(lines[cursor:])
    composed = "".join(result)
    if source_digest(composed) != entry["result_sha256"]:
        raise ValueError(f"PAVG extension verification failed: {relative}")
    return composed


def execute_dependency(relative: str, namespace: dict) -> None:
    source = compose_source(relative)
    # Preserve the public module's path, imports, class names and state-dict keys.
    exec(compile(source, namespace["__file__"], "exec"), namespace)
