"""Bind and acquire the ungated E49-D1 modern-generator diagnostic."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
import pyarrow.parquet as pq

from pixelproof.project_paths import DATA_ROOT, ML_ROOT


NAMESPACE = "E49_D1_DOTTING_V1"
REPO_ID = "fge-auto/dotting-test"
REVISION = "0bcc6877c7d23f4e615b5470f06b1c00e7db7311"
LICENSE = "cc-by-4.0"
METADATA_FILE = "data/parquet/generation_benchmark/0000.parquet"
MODEL_KEYS = {
    "gptimage2": "GPT Image 2",
    "nanobanana2": "Nano Banana 2",
    "flux2pro": "FLUX.2 Pro",
    "ideogram4": "Ideogram 4",
    "seedream5lite": "Seedream 5.0 Lite",
}
AVAILABLE_PER_MODEL = 210
TARGET_PER_MODEL = 160
RESERVE_PER_MODEL = 192
MAX_NETWORK_BYTES = 512 * 1024**2

ROOT = DATA_ROOT / "e49_d1_dotting"
SNAPSHOT = ROOT / "repository"
CONTRACT = ROOT / "source_contract.json"
DOWNLOAD_RECEIPT = ROOT / "download_receipt.json"
CONTRACT_EVIDENCE = ML_ROOT.parent / "evidence" / "e49_d1_dotting_contract.json"
DOWNLOAD_EVIDENCE = ML_ROOT.parent / "evidence" / "e49_d1_dotting_download.json"


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _write_atomic(path: Path, value: Any) -> bytes:
    raw = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(raw)
    temporary.replace(path)
    return raw


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rank(row: Mapping[str, Any]) -> str:
    identity = f"{row['model_key']}|{row['request_id']}|{row['image']}"
    return hashlib.sha256(f"{NAMESPACE}|{identity}".encode()).hexdigest()


def select_reserve(
    metadata_rows: Sequence[Mapping[str, Any]], remote_files: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Select exact model-blind reserves from successful, revision-bound image paths."""
    by_model: dict[str, list[dict[str, Any]]] = {key: [] for key in MODEL_KEYS}
    seen_ids: set[str] = set()
    for row in metadata_rows:
        model_key = str(row.get("model_key", ""))
        image_path = str(row.get("image", ""))
        request_id = str(row.get("request_id", ""))
        if model_key not in by_model or row.get("status") != "ok" or not image_path:
            continue
        if request_id in seen_ids:
            raise ValueError(f"E49-D1 repeated request id: {request_id}")
        seen_ids.add(request_id)
        remote = remote_files.get(image_path)
        if not remote or not image_path.endswith(".webp"):
            raise ValueError(f"E49-D1 image is missing from pinned inventory: {image_path}")
        candidate = {
            "record_id": f"dotting:{request_id}",
            "parent_id": f"dotting:{request_id}",
            "rank": _rank(row),
            "label": 1,
            "source": MODEL_KEYS[model_key],
            "model_key": model_key,
            "request_id": request_id,
            "prompt_key": str(row.get("prompt_key", "")),
            "word_slug": str(row.get("word_slug", "")),
            "image_path": image_path,
            "bytes": int(remote["bytes"]),
            "sha256": str(remote["sha256"]),
            "width": int(row.get("width") or 0),
            "height": int(row.get("height") or 0),
        }
        by_model[model_key].append(candidate)

    output = []
    for model_key, rows in by_model.items():
        if len(rows) != AVAILABLE_PER_MODEL:
            raise ValueError(f"E49-D1 {model_key} population changed: {len(rows)}/{AVAILABLE_PER_MODEL}")
        output.extend(sorted(rows, key=lambda row: (row["rank"], row["record_id"]))[:RESERVE_PER_MODEL])
    if len(output) != RESERVE_PER_MODEL * len(MODEL_KEYS):
        raise ValueError("E49-D1 reserve count changed")
    return sorted(output, key=lambda row: (row["source"], row["rank"], row["record_id"]))


def validate_download(snapshot: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    expected = {str(row["image_path"]): (int(row["bytes"]), str(row["sha256"])) for row in rows}
    for name, (size, sha256) in expected.items():
        path = snapshot / name
        if not path.is_file() or path.stat().st_size != size or _digest(path) != sha256:
            raise ValueError(f"E49-D1 local payload changed: {name}")
    actual = {
        path.relative_to(snapshot).as_posix()
        for path in snapshot.rglob("*")
        if path.is_file()
        and ".cache" not in path.relative_to(snapshot).parts
        and not any(part.startswith("._") for part in path.relative_to(snapshot).parts)
    }
    if actual != set(expected):
        raise ValueError("E49-D1 snapshot contains missing or unexpected payloads")


def _remote_inventory(info: Any) -> dict[str, dict[str, Any]]:
    output = {}
    for item in info.siblings:
        if item.rfilename.startswith("images/"):
            sha256 = getattr(getattr(item, "lfs", None), "sha256", None)
            if not sha256 or item.size is None:
                raise ValueError(f"E49-D1 remote checksum missing: {item.rfilename}")
            output[item.rfilename] = {"bytes": int(item.size), "sha256": str(sha256)}
    return output


def bind() -> dict[str, Any]:
    if CONTRACT.exists() or CONTRACT_EVIDENCE.exists():
        raise FileExistsError("E49-D1 source contract already exists")
    info = HfApi().dataset_info(REPO_ID, revision=REVISION, files_metadata=True)
    if info.sha != REVISION or info.gated is not False:
        raise ValueError("E49-D1 repository identity/access changed")
    if str((info.card_data or {}).get("license", "")).lower() != LICENSE:
        raise ValueError("E49-D1 repository licence changed")
    metadata_path = Path(hf_hub_download(
        repo_id=REPO_ID, repo_type="dataset", revision=REVISION, filename=METADATA_FILE,
    ))
    columns = ["request_id", "status", "word_slug", "model_key", "prompt_key", "image", "width", "height"]
    metadata_rows = pq.read_table(metadata_path, columns=columns).to_pylist()
    if len(metadata_rows) != 8_400:
        raise ValueError(f"E49-D1 metadata row count changed: {len(metadata_rows)}/8400")
    reserve = select_reserve(metadata_rows, _remote_inventory(info))
    total_bytes = sum(row["bytes"] for row in reserve)
    if total_bytes > MAX_NETWORK_BYTES:
        raise ValueError(f"E49-D1 reserve exceeds network stop: {total_bytes}")
    payload = {
        "schema_version": 1,
        "state": "e49_d1_source_frozen_untransferred_unscored",
        "repo_id": REPO_ID,
        "revision": REVISION,
        "license": LICENSE,
        "attribution": "Firat Gelbal and Dotting Test",
        "role": "AI_ONLY_DIAGNOSTIC",
        "models": MODEL_KEYS,
        "counts": {"reserve": len(reserve), "target": TARGET_PER_MODEL * len(MODEL_KEYS),
                   "reserve_per_model": RESERVE_PER_MODEL, "target_per_model": TARGET_PER_MODEL},
        "network": {"expected_image_bytes": total_bytes, "stop_bytes": MAX_NETWORK_BYTES},
        "rows": reserve,
        "forbidden": ["REAL-photo claim", "AUC/balanced-accuracy claim", "Module-1 promotion",
                      "training", "detector-based row replacement", "threshold change"],
        "image_bytes_downloaded": 0,
        "model_scores_created": 0,
    }
    raw = _write_atomic(CONTRACT, payload)
    evidence = {key: value for key, value in payload.items() if key != "rows"}
    evidence.update({
        "contract_sha256": hashlib.sha256(raw).hexdigest(),
        "contract_bytes": len(raw),
        "reserve_identity_sha256": hashlib.sha256(_json_bytes([
            {"record_id": row["record_id"], "sha256": row["sha256"]} for row in reserve
        ])).hexdigest(),
    })
    _write_atomic(CONTRACT_EVIDENCE, evidence)
    return evidence


def _validate_contract() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw = CONTRACT.read_bytes()
    payload = json.loads(raw)
    evidence = json.loads(CONTRACT_EVIDENCE.read_text())
    if (
        payload.get("state") != "e49_d1_source_frozen_untransferred_unscored"
        or payload.get("image_bytes_downloaded") != 0
        or payload.get("model_scores_created") != 0
        or hashlib.sha256(raw).hexdigest() != evidence.get("contract_sha256")
        or payload.get("revision") != REVISION
    ):
        raise ValueError("E49-D1 source contract changed")
    rows = payload.get("rows", [])
    if len(rows) != RESERVE_PER_MODEL * len(MODEL_KEYS):
        raise ValueError("E49-D1 source contract row count changed")
    return payload, rows


def download() -> dict[str, Any]:
    if DOWNLOAD_RECEIPT.exists() or DOWNLOAD_EVIDENCE.exists():
        raise FileExistsError("E49-D1 download receipt already exists")
    contract, rows = _validate_contract()
    names = [str(row["image_path"]) for row in rows]
    expected_bytes = int(contract["network"]["expected_image_bytes"])
    if expected_bytes > MAX_NETWORK_BYTES:
        raise ValueError("E49-D1 frozen transfer exceeds network stop")
    SNAPSHOT.mkdir(parents=True, exist_ok=True)
    resolved = Path(snapshot_download(
        repo_id=REPO_ID, repo_type="dataset", revision=REVISION, local_dir=SNAPSHOT,
        allow_patterns=names, max_workers=8,
    )).resolve()
    if resolved != SNAPSHOT.resolve():
        raise ValueError("E49-D1 downloader resolved an unexpected destination")
    validate_download(SNAPSHOT, rows)
    receipt = {
        "schema_version": 1,
        "state": "e49_d1_reserve_download_complete_unscored",
        "repo_id": REPO_ID,
        "revision": REVISION,
        "license": LICENSE,
        "files": len(rows),
        "bytes": expected_bytes,
        "by_model": dict(sorted(Counter(row["source"] for row in rows).items())),
        "contract_sha256": hashlib.sha256(CONTRACT.read_bytes()).hexdigest(),
        "model_scores_created": 0,
    }
    raw = _write_atomic(DOWNLOAD_RECEIPT, receipt)
    _write_atomic(DOWNLOAD_EVIDENCE, {
        **receipt, "receipt_bytes": len(raw), "receipt_sha256": hashlib.sha256(raw).hexdigest(),
    })
    return receipt


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("bind", "download"))
    args = parser.parse_args(argv)
    result = bind() if args.command == "bind" else download()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
