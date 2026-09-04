"""Bind and acquire the ungated E49-D1 modern-generator diagnostic."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
from io import BytesIO
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from PIL import Image, ImageOps
import pyarrow.parquet as pq

from experiments.e48_manifest import MANIFEST as E48_MANIFEST, _protected_role_hashes
from pixelproof.data_contract import dhash_image
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
PAIRED_ROOT = ROOT / "paired" / "social_q75"
MANIFEST = ROOT / "manifest_unscored.json"
CONTRACT_EVIDENCE = ML_ROOT.parent / "evidence" / "e49_d1_dotting_contract.json"
DOWNLOAD_EVIDENCE = ML_ROOT.parent / "evidence" / "e49_d1_dotting_download.json"
MANIFEST_EVIDENCE = ML_ROOT.parent / "evidence" / "e49_d1_dotting_manifest.json"
MAX_PIXELS = 50_000_000


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


def _decode_bytes(raw: bytes, identity: str) -> dict[str, Any]:
    with Image.open(BytesIO(raw)) as opened:
        opened.verify()
    with Image.open(BytesIO(raw)) as opened:
        width, height = opened.size
        if width <= 0 or height <= 0 or width * height > MAX_PIXELS:
            raise ValueError(f"E49-D1 unsafe image geometry: {identity}")
        opened.load()
        rgb = ImageOps.exif_transpose(opened).convert("RGB")
        return {
            "decoded_format": str(opened.format or "UNKNOWN").upper(),
            "width": width,
            "height": height,
            "mode": str(opened.mode),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "dhash": dhash_image(rgb),
        }


def social_q75_bytes(path: Path) -> bytes:
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        if max(image.size) > 1_080:
            scale = 1_080 / max(image.size)
            image = image.resize(
                (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                Image.Resampling.LANCZOS,
            )
        output = BytesIO()
        image.save(output, format="JPEG", quality=75, subsampling=2, optimize=False)
        return output.getvalue()


def _all_protected_hashes() -> tuple[set[str], set[str], list[dict[str, str]]]:
    exact, dhashes, sources = _protected_role_hashes()
    if not E48_MANIFEST.is_file():
        raise FileNotFoundError("E49-D1 protected E48 manifest missing")
    raw = E48_MANIFEST.read_bytes()
    payload = json.loads(raw)
    for row in payload.get("rows", []):
        if row.get("sha256"):
            exact.add(str(row["sha256"]))
        if row.get("dhash"):
            dhashes.add(str(row["dhash"]))
    sources.append({"path": str(E48_MANIFEST), "sha256": hashlib.sha256(raw).hexdigest()})
    return exact, dhashes, sources


def _audit_originals(
    rows: Sequence[Mapping[str, Any]], prior_exact: set[str], prior_dhash: set[str],
) -> tuple[list[dict[str, Any]], dict[str, list[str]], list[dict[str, str]]]:
    decoded, reasons, failures = [], defaultdict(list), []
    seen_exact: dict[str, str] = {}
    seen_dhash: dict[str, str] = {}
    for row in sorted(rows, key=lambda item: (str(item["rank"]), str(item["record_id"]))):
        record_id = str(row["record_id"])
        path = SNAPSHOT / str(row["image_path"])
        try:
            raw = path.read_bytes()
            found = _decode_bytes(raw, record_id)
            if found["decoded_format"] != "WEBP" or found["sha256"] != row["sha256"]:
                raise ValueError("format or checksum differs from source contract")
            item = {**row, **found, "path": str(path)}
            decoded.append(item)
            if found["sha256"] in prior_exact:
                reasons[record_id].append("protected_exact_overlap")
            if found["dhash"] in prior_dhash:
                reasons[record_id].append("protected_dhash_overlap")
            if found["sha256"] in seen_exact:
                reasons[record_id].append(f"internal_exact_duplicate_of:{seen_exact[found['sha256']]}")
            else:
                seen_exact[found["sha256"]] = record_id
            if found["dhash"] in seen_dhash:
                reasons[record_id].append(f"internal_dhash_duplicate_of:{seen_dhash[found['dhash']]}")
            else:
                seen_dhash[found["dhash"]] = record_id
        except (OSError, ValueError) as error:
            failures.append({"record_id": record_id, "error": f"{type(error).__name__}: {error}"})
            reasons[record_id].append("decode_failure")
    return decoded, dict(sorted(reasons.items())), failures


def freeze_manifest() -> dict[str, Any]:
    if MANIFEST.exists() or MANIFEST_EVIDENCE.exists():
        raise FileExistsError("E49-D1 manifest already exists")
    contract, contract_rows = _validate_contract()
    download_receipt = json.loads(DOWNLOAD_EVIDENCE.read_text())
    if (
        download_receipt.get("state") != "e49_d1_reserve_download_complete_unscored"
        or download_receipt.get("contract_sha256") != hashlib.sha256(CONTRACT.read_bytes()).hexdigest()
        or download_receipt.get("model_scores_created") != 0
    ):
        raise ValueError("E49-D1 download receipt changed")
    validate_download(SNAPSHOT, contract_rows)
    prior_exact, prior_dhash, protected_sources = _all_protected_hashes()
    decoded, reasons, failures = _audit_originals(contract_rows, prior_exact, prior_dhash)

    selected_parents = []
    observations = []
    child_exact: dict[str, str] = {}
    child_dhash: dict[str, str] = {}
    PAIRED_ROOT.mkdir(parents=True, exist_ok=True)
    for model_key, source in MODEL_KEYS.items():
        candidates = sorted(
            (row for row in decoded if row["model_key"] == model_key),
            key=lambda row: (row["rank"], row["record_id"]),
        )
        selected = 0
        for row in candidates:
            record_id = str(row["record_id"])
            if record_id in reasons:
                continue
            child_raw = social_q75_bytes(Path(str(row["path"])))
            child = _decode_bytes(child_raw, record_id + ":social_q75")
            child_id = record_id + ":social_q75"
            child_reasons = []
            if child["sha256"] in prior_exact:
                child_reasons.append("social_protected_exact_overlap")
            if child["dhash"] in prior_dhash:
                child_reasons.append("social_protected_dhash_overlap")
            if child["sha256"] in child_exact:
                child_reasons.append(f"social_internal_exact_duplicate_of:{child_exact[child['sha256']]}")
            if child["dhash"] in child_dhash:
                child_reasons.append(f"social_internal_dhash_duplicate_of:{child_dhash[child['dhash']]}")
            if child_reasons:
                reasons[record_id] = child_reasons
                continue
            child_exact[child["sha256"]] = child_id
            child_dhash[child["dhash"]] = child_id
            destination = PAIRED_ROOT / f"{row['request_id']}.jpg"
            temporary = destination.with_suffix(".jpg.part")
            temporary.write_bytes(child_raw)
            temporary.replace(destination)
            selected_parents.append({
                "parent_id": record_id, "label": 1, "source": source,
                "model_key": model_key, "request_id": row["request_id"],
                "prompt_key": row["prompt_key"], "word_slug": row["word_slug"], "rank": row["rank"],
            })
            observations.extend([
                {**row, "record_id": record_id + ":publisher_original", "parent_id": record_id,
                 "condition": "publisher_original", "status": "unscored"},
                {"record_id": child_id, "parent_id": record_id, "label": 1, "source": source,
                 "model_key": model_key, "request_id": row["request_id"],
                 "prompt_key": row["prompt_key"], "word_slug": row["word_slug"],
                 "rank": row["rank"], "condition": "social_q75", "path": str(destination),
                 "bytes": len(child_raw), **child, "status": "unscored"},
            ])
            selected += 1
            if selected == TARGET_PER_MODEL:
                break
        if selected != TARGET_PER_MODEL:
            raise ValueError(f"E49-D1 cannot fill clean target for {model_key}: {selected}/{TARGET_PER_MODEL}")

    if len(selected_parents) != TARGET_PER_MODEL * len(MODEL_KEYS) or len(observations) != 1_600:
        raise ValueError("E49-D1 frozen target count changed")
    by_source = Counter(row["source"] for row in selected_parents)
    if dict(by_source) != {source: TARGET_PER_MODEL for source in MODEL_KEYS.values()}:
        raise ValueError("E49-D1 frozen target is not model-balanced")
    payload = {
        "schema_version": 1,
        "state": "e49_d1_decontaminated_paired_frozen_unscored",
        "role": "AI_ONLY_DIAGNOSTIC",
        "source_contract_sha256": hashlib.sha256(CONTRACT.read_bytes()).hexdigest(),
        "download_receipt_sha256": hashlib.sha256(DOWNLOAD_RECEIPT.read_bytes()).hexdigest(),
        "counts": {
            "candidates": len(contract_rows), "decoded": len(decoded), "decode_failures": len(failures),
            "identity_exclusions": len(reasons), "parents": len(selected_parents),
            "observations": len(observations), "by_source": dict(sorted(by_source.items())),
        },
        "decode_failures": failures,
        "identity_exclusion_reasons": dict(sorted(reasons.items())),
        "protected_role_manifests": protected_sources,
        "parents": sorted(selected_parents, key=lambda row: (row["source"], row["rank"])),
        "rows": sorted(observations, key=lambda row: (row["condition"], row["source"], row["rank"])),
        "model_scores_created": 0,
        "boundary": "AI-only diagnostic; target and Q75 children frozen before detector access.",
    }
    raw = _write_atomic(MANIFEST, payload)
    evidence = {
        "schema_version": 1, "state": payload["state"], "role": payload["role"],
        "counts": payload["counts"], "source_contract_sha256": payload["source_contract_sha256"],
        "protected_role_manifest_count": len(protected_sources), "manifest_bytes": len(raw),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(), "model_scores_created": 0,
    }
    _write_atomic(MANIFEST_EVIDENCE, evidence)
    return evidence


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
    parser.add_argument("command", choices=("bind", "download", "freeze-manifest"))
    args = parser.parse_args(argv)
    actions = {"bind": bind, "download": download, "freeze-manifest": freeze_manifest}
    result = actions[args.command]()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
