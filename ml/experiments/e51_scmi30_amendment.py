"""Bind and realize score-blind SCMI30 identity-audit replacements for E51 CAL."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import ssl
from typing import Any, Iterable, Mapping, Sequence
import zipfile

import certifi
import fsspec

from experiments.e51_data_route import NAMESPACE, SCMI30_REF, SCMI30_VERSION, _rank
from experiments.e51_train_cal_realize import _decode, _phash_path, _protected
from experiments.e51_transfer import (
    SCMI30_ARCHIVE_BYTES, SCMI30_FILES, SCMI30_RANGE_BLOCK_BYTES,
    _dataset_archive_url, inspect_scmi30_file,
)
from pixelproof.project_paths import DATA_ROOT, ML_ROOT


EXCLUDED_IDENTITY = (
    "kaggle:goyalpuneet/sci30iitrpr:v2:"
    "SCMI30-IITRPR/Similar/D04_Samsung_Galaxy_M04/no_content/D04_black.jpg"
)
RESERVE_COUNT = 5
ROOT = DATA_ROOT / "e51"
ROUTE = ROOT / "route" / "contract_untransferred.json"
INVENTORY = ROOT / "source_audit" / "scmi30_public_inventory.json"
CONTRACT = ROOT / "cal_identity_amendment_unscored.json"
CONTRACT_EVIDENCE = ML_ROOT.parent / "evidence" / "e51_scmi30_amendment_contract.json"
PAYLOAD_ROOT = ROOT / "payloads" / "scmi30_cal_amendment"
STAGING_ROOT = ROOT / "staging" / "scmi30_cal_amendment"
RECEIPT = ROOT / "receipts" / "scmi30_cal_amendment_unscored.json"
EVIDENCE = ML_ROOT.parent / "evidence" / "e51_scmi30_amendment.json"


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _write(path: Path, value: Any) -> bytes:
    raw = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(raw)
    temporary.replace(path)
    return raw


def reserve_rows(
    inventory: Sequence[Mapping[str, Any]], selected_paths: set[str], *, count: int = RESERVE_COUNT,
) -> list[dict[str, Any]]:
    rows = []
    for item in inventory:
        name = str(item["name"])
        parts = name.split("/")
        if (len(parts) < 4 or parts[1] != "Similar"
                or parts[2] != "D04_Samsung_Galaxy_M04" or name in selected_paths
                or not name.lower().endswith((".jpg", ".jpeg"))):
            continue
        identity = f"kaggle:{SCMI30_REF}:v{SCMI30_VERSION}:{name}"
        rows.append({
            "identity": identity, "rank": _rank("SCMI30:D04:Similar", identity),
            "label": 0, "role": "CAL", "source": "SCMI30-IITRPR", "device_id": "D04",
            "device_folder": "D04_Samsung_Galaxy_M04", "branch": "Similar",
            "remote_path": name, "expected_bytes": int(item["total_bytes"]),
        })
    ordered = sorted(rows, key=lambda row: (row["rank"], row["identity"]))
    if len(ordered) < count:
        raise ValueError("insufficient D04 Similar replacement reserve")
    return ordered[:count]


def bind() -> dict[str, Any]:
    if CONTRACT.exists() or CONTRACT_EVIDENCE.exists():
        raise FileExistsError("SCMI30 amendment contract already exists")
    route_raw = ROUTE.read_bytes()
    selected = json.loads(route_raw)["roles"]["cal"]["rows"]
    if EXCLUDED_IDENTITY not in {row["identity"] for row in selected}:
        raise ValueError("confirmed SCMI30 exclusion left the frozen route")
    inventory = json.loads(INVENTORY.read_text())["files"]
    reserves = reserve_rows(inventory, {str(row["remote_path"]) for row in selected})
    payload = {
        "schema_version": 1, "state": "e51_scmi30_identity_amendment_frozen_untransferred_unscored",
        "namespace": NAMESPACE, "route_sha256": hashlib.sha256(route_raw).hexdigest(),
        "excluded": {"identity": EXCLUDED_IDENTITY,
                     "reason": "protected dHash candidate confirmed by pHash distance 0; no_content black frame"},
        "audit": {"parents": 1200, "confirmed_exclusions": 1, "cleared_dhash_candidates": 25,
                  "near_duplicate_phash_distance_max": 4},
        "selection": "first five unselected rows by original namespace rank in same device+branch",
        "reserve_rows": reserves, "new_image_bytes_downloaded": 0, "model_scores_created": 0,
    }
    raw = _write(CONTRACT, payload)
    evidence = {"schema_version": 1, "state": payload["state"], "reserve_rows": len(reserves),
                "reserve_bytes": sum(row["expected_bytes"] for row in reserves),
                "contract_bytes": len(raw), "contract_sha256": hashlib.sha256(raw).hexdigest(),
                "new_image_bytes_downloaded": 0, "model_scores_created": 0}
    _write(CONTRACT_EVIDENCE, evidence)
    return evidence


def download() -> dict[str, Any]:
    if RECEIPT.exists() or EVIDENCE.exists():
        raise FileExistsError("SCMI30 amendment receipt already exists")
    raw = CONTRACT.read_bytes()
    contract_evidence = json.loads(CONTRACT_EVIDENCE.read_text())
    if hashlib.sha256(raw).hexdigest() != contract_evidence.get("contract_sha256"):
        raise ValueError("SCMI30 amendment contract changed")
    rows = json.loads(raw)["reserve_rows"]
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    downloaded = []
    STAGING_ROOT.mkdir(parents=True, exist_ok=True)
    with fsspec.open(_dataset_archive_url(), "rb", block_size=SCMI30_RANGE_BLOCK_BYTES,
                     cache_type="readahead", ssl=ssl_context) as remote:
        with zipfile.ZipFile(remote) as archive:
            infos = archive.infolist()
            if len(infos) != SCMI30_FILES or sum(info.file_size for info in infos) != SCMI30_ARCHIVE_BYTES:
                raise ValueError("SCMI30 amendment archive changed")
            for row in rows:
                temporary = STAGING_ROOT / f"{row['rank']}.part"
                with archive.open(row["remote_path"]) as source, temporary.open("wb") as target:
                    shutil.copyfileobj(source, target, length=1024 * 1024)
                found = inspect_scmi30_file(temporary, row)
                destination = PAYLOAD_ROOT / f"{row['rank']}.jpg"
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary.replace(destination)
                downloaded.append({**found, "path": str(destination)})
    exact, dhashes, _, dhash_paths = _protected()
    clean, rejected = [], []
    for row in downloaded:
        path = Path(row["path"])
        facts, _ = _decode(path.read_bytes(), row["identity"])
        reasons = []
        if facts["sha256"] in exact:
            reasons.append("protected_exact")
        candidates = dhash_paths.get(facts["dhash"], [])
        if facts["dhash"] in dhashes and not candidates:
            reasons.append("unverifiable_dhash")
        if candidates:
            target = _phash_path(path)
            minimum = min((target ^ _phash_path(Path(item))).bit_count() for item in candidates)
            if minimum <= 4:
                reasons.append(f"confirmed_phash:{minimum}")
        if reasons:
            rejected.append({"identity": row["identity"], "reasons": reasons})
        else:
            clean.append({**row, **facts})
    if not clean:
        raise ValueError("SCMI30 amendment has no clean replacement")
    selected = clean[0]
    payload = {"schema_version": 1, "state": "e51_scmi30_identity_amended_unscored",
               "amendment_contract_sha256": contract_evidence["contract_sha256"],
               "excluded_identity": EXCLUDED_IDENTITY, "selected": selected,
               "downloaded_rows": len(downloaded), "downloaded_bytes": sum(r["bytes"] for r in downloaded),
               "rejected": rejected, "model_scores_created": 0}
    receipt_raw = _write(RECEIPT, payload)
    compact = {k: payload[k] for k in ("schema_version", "state", "amendment_contract_sha256",
                                        "excluded_identity", "downloaded_rows", "downloaded_bytes",
                                        "rejected", "model_scores_created")}
    compact.update({"selected_identity": selected["identity"], "receipt_bytes": len(receipt_raw),
                    "receipt_sha256": hashlib.sha256(receipt_raw).hexdigest()})
    _write(EVIDENCE, compact)
    return compact


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("bind", "download"))
    args = parser.parse_args(argv)
    print(json.dumps(bind() if args.command == "bind" else download(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
