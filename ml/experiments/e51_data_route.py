"""Bind E51 TRAIN/CAL/DEVELOPMENT identities before any image payload."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any, Iterable, Mapping, Sequence

from huggingface_hub import HfFileSystem
from kaggle.api.kaggle_api_extended import KaggleApi
import pyarrow.parquet as pq
import requests

from pixelproof.project_paths import DATA_ROOT, ML_ROOT


NAMESPACE = "E51_AUTHENTIC_SAFETY_V1"
ROOT = DATA_ROOT / "e51"
ROUTE_ROOT = ROOT / "route"
CONTRACT = ROUTE_ROOT / "contract_untransferred.json"
EVIDENCE = ML_ROOT.parent / "evidence" / "e51_data_route_contract.json"
SCMI30_CACHE = ROOT / "source_audit" / "scmi30_public_inventory.json"
IEEE_CACHE = ROOT / "source_audit" / "ieee_spcup_inventory.json"

SCMI30_REF = "goyalpuneet/sci30iitrpr"
SCMI30_VERSION = 2
SCMI30_PER_BRANCH_DEVICE = 20

IEEE_REF = "sp-society-camera-model-identification"
IEEE_EXPECTED_FILES = 5_391
IEEE_EXPECTED_TRAIN = 2_750
IEEE_EXPECTED_TEST = 2_640

DATAPOINT_REPO = "datapointai/text-2-image-human-preferences-2m"
DATAPOINT_REVISION = "e1d8719a2d521eac6c62ee84f329afc2c03ec928"
DATAPOINT_CACHE = (
    DATA_ROOT
    / ".hf_cache"
    / "datasets--datapointai--text-2-image-human-preferences-2m"
    / "snapshots"
    / DATAPOINT_REVISION
    / "data"
)
DATAPOINT_MODELS = {
    "flux-2-max": {"display_name": "FLUX.2 [max]", "shards": (1, 2)},
    "gemini-3.1-flash-image": {
        "display_name": "Nano Banana 2", "shards": (5, 6),
    },
    "gpt-image-2-high": {
        "display_name": "GPT Image 2 (high)", "shards": (9,),
    },
    "ideogram-4.0-quality": {
        "display_name": "Fal Ideogram v4.0q", "shards": (13,),
    },
    "seedream-5.0-pro": {
        "display_name": "Seedream 5.0 Pro", "shards": (34,),
    },
}
DATAPOINT_SHARD_BYTES = {
    1: 446_084_644,
    2: 469_874_692,
    5: 472_479_660,
    6: 463_209_404,
    9: 442_428_311,
    13: 464_939_906,
    34: 461_264_976,
}
DATAPOINT_TARGET_PER_MODEL = 160
DATAPOINT_RESERVE_PER_MODEL = 184
DATAPOINT_PROMPT_CATEGORIES = 8
DATAPOINT_RESERVE_PER_CATEGORY = 23


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _write_atomic(path: Path, value: Any) -> bytes:
    raw = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(raw)
    temporary.replace(path)
    return raw


def _rank(source: str, identity: str) -> str:
    return hashlib.sha256(f"{NAMESPACE}|{source}|{identity}".encode()).hexdigest()


def select_scmi30_rows(
    files: Sequence[Mapping[str, Any]], per_branch_device: int = SCMI30_PER_BRANCH_DEVICE,
) -> list[dict[str, Any]]:
    """Freeze equal Random/Similar quotas for every device without reading images."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in files:
        name = str(item["name"])
        if not name.lower().endswith((".jpg", ".jpeg")):
            continue
        parts = name.split("/")
        if len(parts) < 4 or parts[1] not in {"Random", "Similar"}:
            raise ValueError(f"Unexpected SCMI30 image path: {name}")
        branch = parts[1]
        device_folder = parts[2]
        device_id = device_folder.split("_", 1)[0]
        identity = f"kaggle:{SCMI30_REF}:v{SCMI30_VERSION}:{name}"
        grouped[(device_id, branch)].append({
            "identity": identity,
            "rank": _rank(f"SCMI30:{device_id}:{branch}", identity),
            "label": 0,
            "role": "CAL",
            "source": "SCMI30-IITRPR",
            "device_id": device_id,
            "device_folder": device_folder,
            "branch": branch,
            "remote_path": name,
            "expected_bytes": int(item["total_bytes"]),
        })
    devices = sorted({device for device, _ in grouped})
    if len(devices) != 30 or set(grouped) != {
        (device, branch) for device in devices for branch in ("Random", "Similar")
    }:
        raise ValueError("SCMI30 device/branch inventory changed")
    selected: list[dict[str, Any]] = []
    for key in sorted(grouped):
        rows = sorted(grouped[key], key=lambda row: (row["rank"], row["identity"]))
        if len(rows) < per_branch_device:
            raise ValueError(f"SCMI30 quota cannot be filled: {key}")
        selected.extend(rows[:per_branch_device])
    if len(selected) != 30 * 2 * per_branch_device:
        raise ValueError("SCMI30 selected count changed")
    return selected


def summarize_ieee(files: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate the complete IEEE SP Cup inventory and retain the untouched test split."""
    train = [row for row in files if str(row["name"]).startswith("train/train/")]
    test = [row for row in files if str(row["name"]).startswith("test/test/")]
    other = [row for row in files if row not in train and row not in test]
    cells = Counter()
    rows = []
    for item in test:
        name = str(item["name"])
        match = re.fullmatch(r"test/test/(img_[0-9a-f]{7})_(manip|unalt)\.tif", name)
        if not match:
            raise ValueError(f"Unexpected IEEE test path: {name}")
        cell = "postprocessed" if match.group(2) == "manip" else "unaltered"
        cells[cell] += 1
        identity = f"kaggle:{IEEE_REF}:{name}"
        rows.append({
            "identity": identity,
            "label": 0,
            "role": "DEVELOPMENT",
            "source": "IEEE-SP-Cup-2018-test",
            "transport_cell": cell,
            "remote_path": name,
            "expected_bytes": int(item["total_bytes"]),
        })
    if len(files) != IEEE_EXPECTED_FILES or len(train) != IEEE_EXPECTED_TRAIN:
        raise ValueError("IEEE SP Cup full inventory changed")
    if len(rows) != IEEE_EXPECTED_TEST or cells != {"postprocessed": 1_320, "unaltered": 1_320}:
        raise ValueError("IEEE SP Cup test inventory changed")
    if len(other) != 1 or str(other[0]["name"]) != "sample_submission.csv":
        raise ValueError("IEEE SP Cup auxiliary inventory changed")
    return {
        "full_file_count": len(files),
        "full_bytes": sum(int(row["total_bytes"]) for row in files),
        "train_files_excluded": len(train),
        "test_files": len(rows),
        "test_bytes": sum(row["expected_bytes"] for row in rows),
        "transport_cells": dict(sorted(cells.items())),
        "rows": sorted(rows, key=lambda row: row["identity"]),
    }


def _fetch_ieee_inventory() -> list[dict[str, Any]]:
    if IEEE_CACHE.exists():
        cached = json.loads(IEEE_CACHE.read_text())
        if cached.get("competition") != IEEE_REF:
            raise ValueError("IEEE inventory cache identity changed")
        return list(cached["files"])
    api = KaggleApi()
    api.authenticate()
    token = None
    files: list[dict[str, Any]] = []
    while True:
        page = None
        for attempt in range(6):
            try:
                page = api.competition_list_files(IEEE_REF, page_token=token, page_size=200)
                break
            except requests.HTTPError as error:
                if error.response is None or error.response.status_code != 429 or attempt == 5:
                    raise
                time.sleep(min(2.0**attempt, 30.0))
        if page is None:  # pragma: no cover
            raise RuntimeError("IEEE inventory produced no page")
        files.extend({"name": str(row.name), "total_bytes": int(row.total_bytes or 0)}
                     for row in (page.files or []))
        token = page.next_page_token
        if not token:
            break
        time.sleep(0.25)
    _write_atomic(IEEE_CACHE, {
        "schema_version": 1,
        "competition": IEEE_REF,
        "files": files,
    })
    return files


def _prompt_categories() -> dict[int, str]:
    path = DATAPOINT_CACHE / "prompts" / "prompts.parquet"
    rows = pq.read_table(path, columns=["prompt_id", "category"]).to_pylist()
    if len(rows) != 500:
        raise ValueError("Datapoint prompt inventory changed")
    return {index: str(row["category"]) for index, row in enumerate(rows, start=1)}


def select_datapoint_rows(rows: Sequence[Mapping[str, Any]], model_id: str) -> list[dict[str, Any]]:
    """Select a score-blind, exact-SHA reserve from one predeclared transport shard."""
    candidates = []
    seen_sha: set[str] = set()
    for item in rows:
        if str(item["model_id"]) != model_id:
            continue
        sha = str(item["sha256"])
        if sha in seen_sha:
            continue
        seen_sha.add(sha)
        identity = f"hf:{DATAPOINT_REPO}@{DATAPOINT_REVISION}:{item['image_key']}"
        candidates.append({
            "identity": identity,
            "rank": _rank(f"Datapoint:{model_id}", identity),
            "label": 1,
            "role": "DEVELOPMENT",
            "source": str(DATAPOINT_MODELS[model_id]["display_name"]),
            "model_id": model_id,
            "image_key": str(item["image_key"]),
            "prompt_ordinal": int(item["prompt_ordinal"]),
            "prompt_category": str(item["prompt_category"]),
            "width": int(item["width"]),
            "height": int(item["height"]),
            "source_format": str(item["source_format"]),
            "expected_bytes": int(item["byte_size"]),
            "expected_sha256": sha,
            "source_shard": str(item["source_shard"]),
        })
    selected = sorted(candidates, key=lambda row: (row["rank"], row["identity"]))[
        :DATAPOINT_RESERVE_PER_MODEL
    ]
    if len(selected) != DATAPOINT_RESERVE_PER_MODEL:
        raise ValueError(f"Datapoint reserve cannot be filled: {model_id}")
    return selected


def select_datapoint_paired_rows(
    rows_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
) -> tuple[list[dict[str, Any]], list[int]]:
    """Select the same score-blind prompt reserve for every generator."""
    ordinals_by_model = {
        model_id: {int(row["prompt_ordinal"]) for row in rows}
        for model_id, rows in rows_by_model.items()
    }
    if set(ordinals_by_model) != set(DATAPOINT_MODELS):
        raise ValueError("Datapoint model inventory changed")
    common = set.intersection(*ordinals_by_model.values())
    category_by_ordinal = {
        int(row["prompt_ordinal"]): str(row["prompt_category"])
        for row in next(iter(rows_by_model.values()))
    }
    by_category: dict[str, list[int]] = defaultdict(list)
    for ordinal in common:
        by_category[category_by_ordinal[ordinal]].append(ordinal)
    eligible_categories = sorted(
        category for category, ordinals in by_category.items()
        if len(ordinals) >= DATAPOINT_RESERVE_PER_CATEGORY
    )
    if len(eligible_categories) != DATAPOINT_PROMPT_CATEGORIES:
        raise ValueError("Datapoint paired prompt category inventory changed")
    ordered = []
    for category in eligible_categories:
        ranked = sorted(
            by_category[category],
            key=lambda ordinal: (_rank(f"Datapoint:paired_prompt:{category}", str(ordinal)), ordinal),
        )
        ordered.extend(ranked[:DATAPOINT_RESERVE_PER_CATEGORY])
    if len(ordered) != DATAPOINT_RESERVE_PER_MODEL:
        raise ValueError("Datapoint paired prompt reserve cannot be filled")
    selected: list[dict[str, Any]] = []
    for model_id in DATAPOINT_MODELS:
        by_ordinal = {int(row["prompt_ordinal"]): row for row in rows_by_model[model_id]}
        if len(by_ordinal) != len(rows_by_model[model_id]):
            raise ValueError(f"Datapoint duplicate prompt ordinal: {model_id}")
        model_rows = []
        seen_sha: set[str] = set()
        for ordinal in ordered:
            item = by_ordinal[ordinal]
            sha = str(item["sha256"])
            if sha in seen_sha:
                raise ValueError(f"Datapoint duplicate payload in paired reserve: {model_id}")
            seen_sha.add(sha)
            identity = f"hf:{DATAPOINT_REPO}@{DATAPOINT_REVISION}:{item['image_key']}"
            model_rows.append({
                "identity": identity,
                "rank": _rank(f"Datapoint:paired_prompt:{item['prompt_category']}", str(ordinal)),
                "label": 1,
                "role": "DEVELOPMENT",
                "source": str(DATAPOINT_MODELS[model_id]["display_name"]),
                "model_id": model_id,
                "image_key": str(item["image_key"]),
                "prompt_ordinal": ordinal,
                "prompt_category": str(item["prompt_category"]),
                "width": int(item["width"]),
                "height": int(item["height"]),
                "source_format": str(item["source_format"]),
                "expected_bytes": int(item["byte_size"]),
                "expected_sha256": sha,
                "source_shard": str(item["source_shard"]),
            })
        selected.extend(model_rows)
    return selected, ordered


def _datapoint_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    categories = _prompt_categories()
    filesystem = HfFileSystem()
    rows_by_model: dict[str, list[dict[str, Any]]] = {}
    populations: dict[str, int] = {}
    for model_id, spec in DATAPOINT_MODELS.items():
        rows = []
        for shard_value in spec["shards"]:
            shard = int(shard_value)
            source_shard = f"data/images/images-{shard:04d}.parquet"
            remote = f"datasets/{DATAPOINT_REPO}/{source_shard}"
            with filesystem.open(remote, "rb", revision=DATAPOINT_REVISION) as handle:
                table = pq.read_table(handle, columns=[
                    "image_key", "model_id", "width", "height", "source_format", "byte_size", "sha256",
                ])
            for item in table.to_pylist():
                if str(item["model_id"]) != model_id:
                    continue
                match = re.match(r"^[^/]+/(\d{3})-", str(item["image_key"]))
                if not match:
                    raise ValueError(f"Unexpected Datapoint image key: {item['image_key']}")
                ordinal = int(match.group(1))
                item["prompt_ordinal"] = ordinal
                item["prompt_category"] = categories[ordinal]
                item["source_shard"] = source_shard
                rows.append(item)
        populations[model_id] = len(rows)
        rows_by_model[model_id] = rows
    selected, paired_prompt_ordinals = select_datapoint_paired_rows(rows_by_model)
    if len(selected) != len(DATAPOINT_MODELS) * DATAPOINT_RESERVE_PER_MODEL:
        raise ValueError("Datapoint reserve count changed")
    if len({row["expected_sha256"] for row in selected}) != len(selected):
        raise ValueError("Datapoint selected reserve has cross-model duplicate payloads")
    return selected, {
        "eligible_rows_in_bound_shards": populations,
        "source_shards": [f"data/images/images-{index:04d}.parquet"
                           for index in sorted(DATAPOINT_SHARD_BYTES)],
        "source_shard_bytes": sum(DATAPOINT_SHARD_BYTES.values()),
        "selected_payload_bytes": sum(row["expected_bytes"] for row in selected),
        "paired_prompt_count": len(paired_prompt_ordinals),
        "paired_prompt_identity_sha256": hashlib.sha256(
            json.dumps(paired_prompt_ordinals, separators=(",", ":")).encode()
        ).hexdigest(),
    }


def _identity_sha(rows: Sequence[Mapping[str, Any]]) -> str:
    raw = json.dumps([row["identity"] for row in rows], separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def bind() -> dict[str, Any]:
    if CONTRACT.exists() or EVIDENCE.exists():
        raise FileExistsError("E51 data route is already bound")
    scmi_cache = json.loads(SCMI30_CACHE.read_text())
    scmi_rows = select_scmi30_rows(scmi_cache["files"])
    ieee = summarize_ieee(_fetch_ieee_inventory())
    datapoint_rows, datapoint_transport = _datapoint_rows()
    payload = {
        "schema_version": 1,
        "state": "e51_route_frozen_untransferred_unscored",
        "namespace": NAMESPACE,
        "roles": {
            "train": {
                "base": "historical legitimate TRAIN-role parents only",
                "auxiliary": {
                    "source": "SCIMD-17",
                    "zenodo_record": 17317613,
                    "archive_bytes": 174_438_734,
                    "archive_md5": "37da574c9e8d9c0fd3a7c9bedc5d72a6",
                    "restriction": "224x224 resized REAL hard negatives; TRAIN only",
                },
            },
            "cal": {
                "source": "SCMI30-IITRPR",
                "reference": SCMI30_REF,
                "version": SCMI30_VERSION,
                "license": scmi_cache["metadata"]["license_name"],
                "per_device": 40,
                "per_device_branch": SCMI30_PER_BRANCH_DEVICE,
                "rows": scmi_rows,
            },
            "development_real": {
                "source": "IEEE SP Cup 2018",
                "reference": IEEE_REF,
                "publisher": "IEEE Signal Processing Society",
                "access": "awaiting_user_competition_rules_acceptance",
                **ieee,
            },
            "development_ai": {
                "source": "Datapoint text-2-image-human-preferences-2m",
                "repo": DATAPOINT_REPO,
                "revision": DATAPOINT_REVISION,
                "license": "cc-by-4.0",
                "target_per_model_after_realization": DATAPOINT_TARGET_PER_MODEL,
                "reserve_per_model": DATAPOINT_RESERVE_PER_MODEL,
                "transport_policy": (
                    "Seven predeclared Parquet shards stay below the 4 GiB transfer budget; the "
                    "same 23 score-blind prompts in each of eight categories are selected for all "
                    "five generators so comparisons cannot inherit a content confound."
                ),
                **datapoint_transport,
                "rows": datapoint_rows,
            },
        },
        "independence": {
            "e51_development_forbidden_for": ["TRAIN", "CAL", "E52 final"],
            "e51_cal_forbidden_for": ["TRAIN", "DEVELOPMENT", "E52 final"],
            "required_before_admission": [
                "exact SHA-256 and decode receipt",
                "protected-role SHA-256 and dHash audit",
                "parent-grouped original plus deterministic Q75 pairing",
            ],
        },
        "forbidden": [
            "model access before realized manifests are frozen",
            "score-dependent row replacement",
            "threshold fitting outside CAL",
            "claiming IEEE/NIST/Google certification",
        ],
        "new_image_bytes_downloaded": 0,
        "model_scores_created": 0,
    }
    raw = _write_atomic(CONTRACT, payload)
    evidence = {
        "schema_version": 1,
        "state": payload["state"],
        "contract_bytes": len(raw),
        "contract_sha256": hashlib.sha256(raw).hexdigest(),
        "train_auxiliary_archive_bytes": 174_438_734,
        "cal": {
            "parents": len(scmi_rows),
            "devices": len({row["device_id"] for row in scmi_rows}),
            "branches": dict(sorted(Counter(row["branch"] for row in scmi_rows).items())),
            "expected_bytes": sum(row["expected_bytes"] for row in scmi_rows),
            "identity_sha256": _identity_sha(scmi_rows),
        },
        "development_real": {
            key: ieee[key] for key in (
                "full_file_count", "full_bytes", "test_files", "test_bytes", "transport_cells"
            )
        } | {"identity_sha256": _identity_sha(ieee["rows"]),
             "access": "awaiting_user_competition_rules_acceptance"},
        "development_ai": {
            "models": len(DATAPOINT_MODELS),
            "reserve_rows": len(datapoint_rows),
            "target_rows_after_realization": len(DATAPOINT_MODELS) * DATAPOINT_TARGET_PER_MODEL,
            "source_shard_bytes": datapoint_transport["source_shard_bytes"],
            "selected_payload_bytes": datapoint_transport["selected_payload_bytes"],
            "paired_prompt_count": datapoint_transport["paired_prompt_count"],
            "paired_prompt_identity_sha256": datapoint_transport["paired_prompt_identity_sha256"],
            "identity_sha256": _identity_sha(datapoint_rows),
            "prompt_categories_by_model": {
                model: dict(sorted(Counter(
                    row["prompt_category"] for row in datapoint_rows if row["model_id"] == model
                ).items())) for model in DATAPOINT_MODELS
            },
        },
        "new_image_bytes_downloaded": 0,
        "model_scores_created": 0,
    }
    _write_atomic(EVIDENCE, evidence)
    return evidence


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("bind",))
    args = parser.parse_args(argv)
    if args.command == "bind":
        print(json.dumps(bind(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
