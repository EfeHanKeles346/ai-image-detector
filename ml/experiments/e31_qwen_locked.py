"""E31/B5 one-shot Qwen LOCKED FINAL scout; requires committed DEVELOPMENT pass evidence."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from PIL import Image
from scipy.stats import beta

from pixelproof.data_contract import DataRole, load_manifest, sha256_bytes
from pixelproof.e31_candidate import CANDIDATE_SHA256, E31Candidate
from pixelproof.project_paths import DATA_ROOT

DATASET_DIR = DATA_ROOT / "e30/qwen_locked_final"
MANIFEST_PATH = DATASET_DIR / "derived_manifest.json"
OUTPUT_PATH = DATA_ROOT / "e30/qwen_locked_final/e31_scores.jsonl"
RESULT_PATH = DATA_ROOT / "e30/qwen_locked_final/e31_result.json"
EXPECTED_CONTENT_SET = "93dcbc01e517eaa61e693c4753a72e8d69136b0105c9c36cb8353c6ad98b749c"
EXPECTED_SELECTION = "50e3fec166c900365145854bfe5183764bbb8d655149d81c524dcbff18901eeb"


def exact_interval(successes: int, total: int, alpha: float = 0.05) -> list[float]:
    lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, total - successes + 1))
    upper = 1.0 if successes == total else float(beta.ppf(1 - alpha / 2, successes + 1, total - successes))
    return [lower, upper]


def rate(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    values = list(rows)
    hits = sum(bool(row["predicted_ai"]) for row in values)
    return {
        "n": len(values),
        "ai_triggers": hits,
        "recall": hits / len(values),
        "exact_95_ci": exact_interval(hits, len(values)),
    }


def summarize(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    failures = [row for row in rows if row["status"] != "ok"]
    ok = [row for row in rows if row["status"] == "ok"]
    by_transport: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_generator: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_transport_generator: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in ok:
        by_transport[row["transport"]].append(row)
        if row["transport"] == "native_source":
            by_generator[row["generator"]].append(row)
        by_transport_generator[f"{row['transport']}::{row['generator']}"].append(row)
    transports = {name: rate(values) for name, values in sorted(by_transport.items())}
    native = transports["native_source"]["recall"]
    standardized = transports["standardized_jpeg"]["recall"]
    return {
        "accounting": {"expected": len(rows), "succeeded": len(ok), "failed": len(failures)},
        "native_recall": transports["native_source"],
        "standardized_recall": transports["standardized_jpeg"],
        "standardized_minus_native": standardized - native,
        "native_per_generator": {
            name: rate(values) for name, values in sorted(by_generator.items())
        },
        "per_transport_generator": {
            name: rate(values) for name, values in sorted(by_transport_generator.items())
        },
        "failures": failures,
        "claim_boundary": "Five native images per generator are diagnostic only; transports repeat the same 40 prompts.",
    }


def load_development_gate(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("state") != "development_passed":
        raise PermissionError("Qwen LOCKED FINAL requires a committed E31 DEVELOPMENT pass")
    if payload.get("candidate_sha256") != CANDIDATE_SHA256:
        raise PermissionError("DEVELOPMENT evidence belongs to another candidate")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--development-evidence", type=Path, default=Path("evidence/e31_b5_development.json")
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    args = parser.parse_args()
    gate = load_development_gate(args.development_evidence)
    manifest, rows = load_manifest(MANIFEST_PATH, require_hashes=True)
    if manifest.get("content_set_sha256") != EXPECTED_CONTENT_SET:
        raise RuntimeError("Qwen LOCKED FINAL content set changed")
    if len(rows) != 80 or {row.role for row in rows} != {DataRole.LOCKED_FINAL_TEST}:
        raise RuntimeError("Qwen LOCKED FINAL must contain 40 parents plus 40 derivatives")
    parents = [row for row in rows if row.parent_id is None]
    if len(parents) != 40 or len({row.generator for row in parents}) != 8:
        raise RuntimeError("Qwen native scout contract changed")
    device = torch.device(
        "mps" if args.device == "auto" and torch.backends.mps.is_available() else
        "cpu" if args.device == "auto" else args.device
    )
    candidate = E31Candidate(device=device)
    contract = candidate.contract()
    contract_sha = sha256_bytes(json.dumps(contract, sort_keys=True).encode())
    cached: dict[str, dict[str, Any]] = {}
    if OUTPUT_PATH.is_file():
        for line in OUTPUT_PATH.read_text().splitlines():
            row = json.loads(line)
            if row["contract_sha256"] != contract_sha:
                raise RuntimeError("Qwen score cache belongs to another candidate")
            cached[row["record_id"]] = row
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("a") as stream:
        for index, record in enumerate(rows, 1):
            if record.record_id in cached:
                continue
            path = DATASET_DIR / record.path
            raw = path.read_bytes()
            base = {
                "record_id": record.record_id,
                "parent_id": record.parent_id,
                "content_id": record.content_id,
                "generator": record.generator,
                "transport": record.transport,
                "contract_sha256": contract_sha,
            }
            try:
                if sha256_bytes(raw) != record.sha256:
                    raise RuntimeError("Qwen image hash changed")
                with Image.open(path) as opened:
                    image = opened.convert("RGB")
                result = candidate.score_image(image, record.content_id or record.record_id)
                output = {
                    **base,
                    "status": "ok",
                    "score": result.score,
                    "predicted_ai": result.predicted_ai,
                    "error": None,
                }
            except Exception as error:
                output = {
                    **base,
                    "status": "error",
                    "score": None,
                    "predicted_ai": False,
                    "error": f"{type(error).__name__}: {error}"[:300],
                }
            stream.write(json.dumps(output, separators=(",", ":")) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            cached[record.record_id] = output
            if index % 10 == 0 or index == len(rows):
                print(f"qwen: {index}/80 accounted", flush=True)
    ordered = [cached[row.record_id] for row in rows]
    payload = {
        "schema_version": 1,
        "experiment": "E31/B5-Qwen-LOCKED",
        "state": "locked_final_scored_once",
        "role": "locked_final_test",
        "selection_sha256": EXPECTED_SELECTION,
        "content_set_sha256": EXPECTED_CONTENT_SET,
        "candidate_contract": contract,
        "candidate_contract_sha256": contract_sha,
        "development_evidence_sha256": sha256_bytes(args.development_evidence.read_bytes()),
        "development_gate": gate["state"],
        "summary": summarize(ordered),
    }
    temporary = RESULT_PATH.with_suffix(".json.part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(RESULT_PATH)
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
