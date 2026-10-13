"""Audit the distributable tree without loading models or datasets."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".md", ".txt", ".yaml", ".yml", ".json", ".toml", ".sh"}
PRIVATE_PATH = re.compile(
    r"(?<![\w])(?:[A-Za-z]:[\\/]|/(?:mnt|home|root|data1|Users|opt)/"
    r"|(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?)"
)
SECRET = re.compile(r"sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9_]{20,}|github_pat_[A-Za-z0-9_]{20,}|BEGIN (?:RSA |OPENSSH )?PRIVATE KEY")
EXCLUDED = {".git", "__pycache__", ".pytest_cache", ".venv", "outputs", "runs",
            "results", "datasets", "weights", "manifests", "metadata", "cache", "third_party"}
BINARY_BLOCKLIST = {".h5", ".hdf5", ".pt", ".pth", ".safetensors", ".ckpt",
                    ".log", ".pyc", ".so", ".dll", ".o"}


def audit(root=ROOT):
    findings = []
    records = []
    for path in sorted(Path(root).rglob("*")):
        relative = path.relative_to(root)
        if any(part in EXCLUDED for part in relative.parts):
            continue
        if relative.parts[:2] == ("configs", "local"):
            continue
        if not path.is_file():
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(Path(root).resolve()):
            findings.append(f"{relative}: linked source outside the distributable tree")
        if path.suffix in BINARY_BLOCKLIST:
            findings.append(f"{relative}: generated or private artifact")
        if path.stat().st_size > 10 * 1024 * 1024:
            findings.append(f"{relative}: exceeds 10 MiB source/example limit")
        data = path.read_bytes()
        records.append({"path": relative.as_posix(), "bytes": len(data),
                        "sha256": hashlib.sha256(data).hexdigest()})
        if path.suffix in TEXT_SUFFIXES:
            text = data.decode("utf-8-sig")
            for number, line in enumerate(text.splitlines(), 1):
                package_pin = path.name == "requirements.txt" and re.fullmatch(
                    r"[A-Za-z0-9_.-]+==[0-9]+(?:\.[0-9]+)+", line.strip()
                )
                if PRIVATE_PATH.search(line) and not package_pin:
                    findings.append(f"{relative}:{number}: fixed machine address")
                if SECRET.search(line):
                    findings.append(f"{relative}:{number}: possible secret")
            if path.suffix == ".py":
                try:
                    ast.parse(text, filename=str(relative))
                except SyntaxError as exc:
                    findings.append(f"{relative}:{exc.lineno}: {exc.msg}")
        if path.suffix == ".md":
            text = data.decode("utf-8-sig")
            for target in re.findall(r"\[[^\]]+\]\(([^)]+)\)", text):
                if "://" in target or target.startswith("#"):
                    continue
                if not (path.parent / target.split("#")[0]).exists():
                    findings.append(f"{relative}: broken link {target}")
    return {"files": records, "findings": findings, "passed": not findings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"files": len(report["files"]), "passed": report["passed"],
                      "findings": report["findings"]}, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
