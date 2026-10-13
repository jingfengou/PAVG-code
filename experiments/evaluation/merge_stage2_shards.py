"""Merge full-manifest Stage 2 shards after strict provenance and coverage checks."""

import argparse
import json
from pathlib import Path

from fixed_stage2_results import merge_shards, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-manifest", required=True, type=Path)
    parser.add_argument("--shard", required=True, action="append", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    inputs = {path.resolve() for path in [args.test_manifest, *args.shard]}
    if args.output.resolve() in inputs:
        raise ValueError("Output must not overwrite the manifest or an input shard")
    result = merge_shards(args.shard, args.test_manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8",
    )
    args.output.with_suffix(args.output.suffix + ".sha256").write_text(
        f"{sha256(args.output)}  {args.output.name}\n", encoding="utf-8",
    )
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
