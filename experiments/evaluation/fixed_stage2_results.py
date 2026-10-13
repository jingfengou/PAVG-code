"""Validation and sample-level aggregation for fixed Stage 2 results."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


FIXED_COUNTS = {"solid": 500, "gas": 100, "interaction": 100}
REQUIRED_METRICS = {"vIoU", "CD", "Corr"}
OVERALL_AGGREGATION = "0.25 * solid + 0.25 * gas + 0.50 * interaction"
IDENTITY_KEYS = (
    "checkpoint_sha256", "config_sha256", "manifest_sha256",
    "evaluator_sha256", "metrics_sha256", "aggregation_sha256",
    "num_inference_steps", "seed", "seed_policy",
    "cross_attention_enabled", "metric_protocol",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_metrics(values, label):
    if not isinstance(values, dict) or not REQUIRED_METRICS.issubset(values):
        raise ValueError(f"Missing metrics for {label}")
    if not all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(value) for value in values.values()
    ):
        raise ValueError(f"Non-finite or nonnumeric metrics for {label}")


def sample_group(row):
    presence = row.get("phase_presence")
    if not isinstance(presence, list) or len(presence) != 2:
        raise ValueError("phase_presence must contain [gas, solid]")
    if any(type(value) is not bool for value in presence) or not any(presence):
        raise ValueError("Invalid phase_presence")
    gas, solid = presence
    for active, key in ((gas, "gas"), (solid, "object"), (gas and solid, "interaction")):
        if active:
            _check_metrics(row.get(key), f"index={row.get('index')} {key}")
        elif row.get(key) is not None:
            raise ValueError(f"Unexpected inactive {key} metrics")
    if gas and solid:
        for key, stream in (("interaction_gas", "gas"), ("interaction_solid", "object")):
            _check_metrics(row.get(key), key)
            if row[key] != row[stream]:
                raise ValueError(f"{key} disagrees with its active stream")
        return "interaction"
    return "gas" if gas else "solid"


def _mean(records):
    if not records:
        return {}
    keys = set(records[0])
    if any(set(record) != keys for record in records):
        raise ValueError("Inconsistent metric keys within a group")
    return {
        key: math.fsum(record[key] for record in records) / len(records)
        for key in sorted(keys)
    }


def summarize_rows(rows, expected_counts=FIXED_COUNTS):
    groups = {key: [] for key in ("solid", "gas", "interaction",
                                  "interaction_solid", "interaction_gas")}
    seen = set()
    for row in rows:
        index = row.get("index")
        if type(index) is not int or index < 0 or index in seen:
            raise ValueError("Invalid or duplicate manifest index")
        seen.add(index)
        group = sample_group(row)
        metric_key = {"solid": "object", "gas": "gas", "interaction": "interaction"}[group]
        groups[group].append(row[metric_key])
        if group == "interaction":
            for key in ("interaction_solid", "interaction_gas"):
                groups[key].append(row[key])
    counts = {key: len(groups[key]) for key in FIXED_COUNTS}
    values = {key: _mean(records) for key, records in groups.items()}
    complete = counts == expected_counts and seen == set(range(sum(expected_counts.values())))
    overall = None
    if complete:
        common = set(values["solid"]) & set(values["gas"]) & set(values["interaction"])
        overall = {
            key: 0.25 * values["solid"][key] + 0.25 * values["gas"][key]
            + 0.50 * values["interaction"][key]
            for key in sorted(common)
        }
    return {
        "group_counts": counts,
        "complete_fixed_test": complete,
        "overall_aggregation": OVERALL_AGGREGATION,
        "metrics": {
            **values, "overall": overall,
            "solid_stream_diagnostic": values["solid"],
            "gas_stream_diagnostic": values["gas"],
            "interaction_joint": values["interaction"],
            "interaction_solid_stream": values["interaction_solid"],
            "interaction_gas_stream": values["interaction_gas"],
        },
    }


def validate_shard(payload, manifest):
    bounds = payload.get("sample_range")
    if not isinstance(bounds, list) or len(bounds) != 2:
        raise ValueError("Missing shard range")
    start, end = bounds
    if any(type(value) is not int for value in bounds) or not 0 <= start < end <= len(manifest):
        raise ValueError("Invalid shard range")
    rows = payload.get("per_sample", [])
    if payload.get("num_samples") != end - start or len(rows) != end - start:
        raise ValueError("Incorrect shard sample count")
    if [row.get("index") for row in rows] != list(range(start, end)):
        raise ValueError("Missing, duplicated, or unordered manifest indices")
    if payload.get("seed_index_offset") != 0:
        raise ValueError("Merge requires shards from the full manifest, not sliced manifests")
    if payload.get("seed_policy") != "seed_plus_seed_index_offset_plus_manifest_index":
        raise ValueError("Unsupported Stage 2 seed policy")
    if payload.get("seed") != 20260804 or payload.get("num_inference_steps") != 25:
        raise ValueError("Formal Stage 2 evaluation requires seed 20260804 and DDIM25")
    for row in rows:
        if row.get("source") != manifest[row["index"]]:
            raise ValueError(f"Source differs from manifest at index {row['index']}")
        if row.get("seed") != payload["seed"] + row["index"]:
            raise ValueError(f"Incorrect sample seed at index {row['index']}")
        sample_group(row)
    return rows


def merge_shards(shards, manifest_path, expected_counts=FIXED_COUNTS):
    if not shards:
        raise ValueError("At least one shard is required")
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, list) or not all(isinstance(value, str) for value in manifest):
        raise ValueError("Manifest must be an array of sample paths")
    if len(manifest) != sum(expected_counts.values()) or len(set(manifest)) != len(manifest):
        raise ValueError("Manifest count or unique sample membership is invalid")
    manifest_hash = sha256(manifest_path)
    reference = None
    rows, provenance = [], []
    for path in map(Path, shards):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if any(key not in payload or payload[key] is None for key in IDENTITY_KEYS):
            raise ValueError(f"Missing provenance: {path}")
        if payload["manifest_sha256"] != manifest_hash:
            raise ValueError(f"Manifest hash mismatch: {path}")
        for key in IDENTITY_KEYS:
            if key.endswith("_sha256"):
                value = payload[key]
                if (not isinstance(value, str) or len(value) != 64
                        or any(char not in "0123456789abcdef" for char in value)):
                    raise ValueError(f"Invalid {key}: {path}")
        if type(payload["cross_attention_enabled"]) is not bool:
            raise ValueError("Missing cross-attention state")
        if reference is None:
            reference = payload
        for key in IDENTITY_KEYS:
            if payload[key] != reference[key]:
                raise ValueError(f"Shards disagree on {key}: {path}")
        rows.extend(validate_shard(payload, manifest))
        record = {"path": str(path), "sha256": sha256(path)}
        if path.with_suffix(".npz").is_file():
            record["predictions_sha256"] = sha256(path.with_suffix(".npz"))
        provenance.append(record)
    rows.sort(key=lambda row: row["index"])
    if [row["index"] for row in rows] != list(range(len(manifest))):
        raise ValueError("Shards must cover the complete manifest exactly once")
    summary = summarize_rows(rows, expected_counts)
    if not summary["complete_fixed_test"]:
        raise ValueError(f"Incorrect phase counts: {summary['group_counts']}")
    result = {key: reference[key] for key in IDENTITY_KEYS}
    result.update(
        method="PAVG fixed Stage 2 merged evaluator",
        checkpoint=reference.get("checkpoint"), config=reference.get("config"),
        manifest=str(manifest_path), split="fixed source-disjoint full test",
        num_samples=len(rows), sample_range=[0, len(rows)],
        seed_index_offset=0, shards=provenance, per_sample=rows, **summary,
    )
    return result
