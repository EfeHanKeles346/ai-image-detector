"""Append iPhone 14 parents to the exact frozen R0 standardized-input contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pixelproof.project_paths import DATA_ROOT, ML_ROOT


EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

import e32_r0_input as r0  # noqa: E402


E32_ROOT = DATA_ROOT / "e32"
MANIFEST = E32_ROOT / "r1b_role_manifest.json"
MANIFEST_EVIDENCE = ML_ROOT.parent / "evidence" / "e32_r1b_role_manifest.json"
R0_RECEIPT = E32_ROOT / "r0_input_receipt.json"
R0_EVIDENCE = ML_ROOT.parent / "evidence" / "e32_r0_input_receipt.json"
OUTPUT = E32_ROOT / "r1b_input_receipt.json"
EVIDENCE = ML_ROOT.parent / "evidence" / "e32_r1b_input_receipt.json"
WORKERS = 6


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write_atomic(path: Path, value: Any) -> bytes:
    raw = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(raw)
    temporary.replace(path)
    return raw


def merge_receipts(
    old_rows: Sequence[Mapping[str, Any]], new_rows: Sequence[Mapping[str, Any]], manifest_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    merged = [dict(row) for row in old_rows] + [dict(row) for row in new_rows]
    if len({row["record_id"] for row in merged}) != len(merged):
        raise ValueError("input receipt record id collision")
    by_id = {str(row["record_id"]): row for row in manifest_rows}
    if set(by_id) != {str(row["record_id"]) for row in merged}:
        raise ValueError("input receipt and R1b manifest differ")
    for row in merged:
        manifest = by_id[str(row["record_id"])]
        if (row["source_id"], row["label"], row["role"]) != (
            manifest["source_id"], manifest["label"], manifest["role"]
        ):
            raise ValueError(f"input receipt metadata changed for {row['record_id']}")
    return sorted(merged, key=lambda row: str(row["record_id"]))


def realize() -> dict[str, Any]:
    manifest_raw = MANIFEST.read_bytes()
    manifest_compact = json.loads(MANIFEST_EVIDENCE.read_text())
    if manifest_compact.get("detailed_report_sha256") != _sha256(manifest_raw):
        raise ValueError("R1b manifest binding changed")
    manifest = json.loads(manifest_raw)
    old_raw = R0_RECEIPT.read_bytes()
    old_compact = json.loads(R0_EVIDENCE.read_text())
    if old_compact.get("detailed_report_sha256") != _sha256(old_raw):
        raise ValueError("R0 input receipt binding changed")
    old = json.loads(old_raw)
    new_manifest = [row for row in manifest["records"] if row["source_id"] == "csafe-mcsidb-iphone14"]
    if len(new_manifest) != 3_994:
        raise ValueError("R1b iPhone manifest count changed")

    realized = []

    def one(record: Mapping[str, Any]) -> dict[str, Any]:
        path = E32_ROOT / str(record["source_key"])
        return r0._materialize(record, path.read_bytes())

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = [pool.submit(one, row) for row in new_manifest]
        for index, future in enumerate(as_completed(futures), start=1):
            realized.append(future.result())
            if index % 250 == 0 or index == len(futures):
                print(f"R1b iPhone standardized {index}/{len(futures)}", flush=True)
    rows = merge_receipts(old["records"], realized, manifest["records"])
    if len(rows) != 26_682:
        raise ValueError("R1b standardized input count changed")
    report = {
        "schema_version": 1,
        "experiment": "E32/C4-R1b-standardized-input-extension",
        "state": "r1b_input_realization_complete",
        "manifest_sha256": _sha256(manifest_raw),
        "r0_receipt_sha256": old_compact["detailed_report_sha256"],
        "controlled_change": "reuse all R0 derived bytes; append iPhone14 through identical transform",
        "preprocessing": old["preprocessing"],
        "record_count": len(rows),
        "appended_records": len(realized),
        "class_counts": dict(sorted(Counter(row["label"] for row in rows).items())),
        "role_counts": dict(sorted(Counter(row["role"] for row in rows).items())),
        "source_counts": dict(sorted(Counter(row["source_id"] for row in rows).items())),
        "total_input_bytes": sum(int(row["input_bytes"]) for row in rows),
        "records_sha256": _sha256(_json_bytes(rows)),
        "records": rows,
        "boundary": "Only TRAIN/CALIBRATION bytes realized; IPN and owner gallery remain unread by preprocessing.",
    }
    raw = _write_atomic(OUTPUT, report)
    compact = {key: value for key, value in report.items() if key != "records"}
    compact.update(
        {
            "detailed_report_bytes": len(raw),
            "detailed_report_sha256": _sha256(raw),
            "detailed_report_external_path": str(OUTPUT.relative_to(E32_ROOT)),
        }
    )
    _write_atomic(EVIDENCE, compact)
    return compact


def main(argv: Iterable[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__).parse_args(argv)
    print(json.dumps(realize(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
