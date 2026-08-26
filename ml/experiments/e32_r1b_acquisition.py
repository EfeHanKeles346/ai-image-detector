"""Freeze and acquire the E32/R1b authentic correction sources.

The command deliberately separates metadata freezing from selected-byte transfer::

    PIXELPROOF_DATA_ROOT=/Volumes/LaCie/pixelproof-datasets \
      PYTHONPATH=ml/src ml/.venv/bin/python ml/experiments/e32_r1b_acquisition.py freeze
    PIXELPROOF_DATA_ROOT=/Volumes/LaCie/pixelproof-datasets \
      PYTHONPATH=ml/src ml/.venv/bin/python ml/experiments/e32_r1b_acquisition.py download-ipn
    PIXELPROOF_DATA_ROOT=/Volumes/LaCie/pixelproof-datasets \
      PYTHONPATH=ml/src ml/.venv/bin/python ml/experiments/e32_r1b_acquisition.py download-csafe
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable, Mapping

import requests

from pixelproof.project_paths import DATA_ROOT, ML_ROOT


OUTPUT_ROOT = DATA_ROOT / "e32"
SELECTION = OUTPUT_ROOT / "r1b_acquisition_selection.json"
EVIDENCE = ML_ROOT.parent / "evidence" / "e32_r1b_acquisition_selection.json"
IPN_RECEIPT = OUTPUT_ROOT / "r1b_ipn_download_receipt.json"
CSAFE_RECEIPT = OUTPUT_ROOT / "r1b_csafe_iphone14_download_receipt.json"
FIGSHARE_API = "https://api.figshare.com/v2/articles/{article_id}"
MIN_FREE_BYTES = 100 * 1024**3
IPN_EXPECTED_COUNT = 960
IPN_EXPECTED_BYTES = 3_889_897_594
IPN_UMBRELLA = {"id": 25201319, "version": 3, "license": "CC BY 4.0"}
IPN_ARTICLES = (
    (25137734, "iphone-se-2020-1"),
    (25137035, "iphone-xr"),
    (25138100, "motorola-g4-plus"),
    (25138133, "samsung-galaxy-a01"),
    (25139678, "samsung-galaxy-note9"),
    (25139681, "motorola-g-play"),
    (25149656, "motorola-g20"),
    (25149680, "iphone-se-2020-2"),
    (25149689, "sony-xperia-m4"),
    (25149713, "huawei-p20-lite"),
    (25149722, "huawei-y9-2019"),
    (25149953, "lg-l65"),
)
CSAFE = {
    "article_id": 26932084,
    "version": 1,
    "license": "CC BY 4.0",
    "file_id": 49389769,
    "name": "iPhone14.zip",
    "bytes": 20_428_338_922,
    "md5": "dfc01c89b14356141f53d253b72e946c",
}
MD5_RE = re.compile(r"^[0-9a-f]{32}$")
CONTENT_RANGE_RE = re.compile(r"^bytes (\d+)-(\d+)/(\d+)$")
CSAFE_RANGE_WORKERS = 4
RETRY_DELAYS = (0, 5, 20, 60, 180)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _write_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(_json_bytes(value))
    temporary.replace(path)


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _md5(path: Path) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def _article(article_id: int) -> dict[str, Any]:
    response = requests.get(FIGSHARE_API.format(article_id=article_id), timeout=(20, 120))
    response.raise_for_status()
    return response.json()


def _validate_article(payload: Mapping[str, Any], expected: Mapping[str, Any]) -> None:
    for field in ("id", "version"):
        if int(payload.get(field, -1)) != int(expected[field]):
            raise ValueError(f"Figshare {field} changed for article {expected['id']}")
    if payload.get("license", {}).get("name") != expected["license"]:
        raise ValueError(f"Figshare licence changed for article {expected['id']}")


def select_ipn_natural(payload: Mapping[str, Any], article_id: int, device: str) -> list[dict[str, Any]]:
    _validate_article(payload, {"id": article_id, "version": 1, "license": "CC BY 4.0"})
    selected = []
    for item in payload.get("files", []):
        name = str(item.get("name", ""))
        if " natural " not in name.lower() or Path(name).suffix.lower() not in {".jpg", ".jpeg"}:
            continue
        supplied = str(item.get("supplied_md5", "")).lower()
        computed = str(item.get("computed_md5", supplied)).lower()
        if supplied != computed or not MD5_RE.fullmatch(supplied):
            raise ValueError(f"IPN checksum contract invalid: {article_id}/{name}")
        size = int(item.get("size", -1))
        if size <= 0 or not str(item.get("download_url", "")).startswith("https://"):
            raise ValueError(f"IPN transfer contract invalid: {article_id}/{name}")
        selected.append(
            {
                "article_id": article_id,
                "article_version": 1,
                "device": device,
                "file_id": int(item["id"]),
                "name": name,
                "bytes": size,
                "md5": supplied,
                "url": str(item["download_url"]),
                "role": "development",
                "label": "real",
            }
        )
    selected.sort(key=lambda row: row["file_id"])
    if len(selected) != 80:
        raise ValueError(f"IPN article {article_id} expected 80 natural JPEGs, found {len(selected)}")
    if len({row["file_id"] for row in selected}) != len(selected):
        raise ValueError(f"IPN article {article_id} repeats a file id")
    return selected


def select_csafe(payload: Mapping[str, Any]) -> dict[str, Any]:
    _validate_article(
        payload,
        {"id": CSAFE["article_id"], "version": CSAFE["version"], "license": CSAFE["license"]},
    )
    item = next(
        (row for row in payload.get("files", []) if int(row.get("id", -1)) == CSAFE["file_id"]),
        None,
    )
    if item is None:
        raise ValueError("CSAFE iPhone14 archive disappeared")
    expected = (CSAFE["name"], CSAFE["bytes"], CSAFE["md5"])
    found = (item.get("name"), int(item.get("size", -1)), item.get("computed_md5"))
    if found != expected:
        raise ValueError(f"CSAFE iPhone14 contract changed: expected {expected}, found {found}")
    if not str(item.get("download_url", "")).startswith("https://"):
        raise ValueError("CSAFE iPhone14 URL is not HTTPS")
    return {**CSAFE, "url": str(item["download_url"]), "role": "train_cal_candidate", "label": "real"}


def freeze() -> dict[str, Any]:
    umbrella = _article(IPN_UMBRELLA["id"])
    _validate_article(umbrella, IPN_UMBRELLA)
    ipn_rows = []
    for article_id, device in IPN_ARTICLES:
        ipn_rows.extend(select_ipn_natural(_article(article_id), article_id, device))
    if len(ipn_rows) != IPN_EXPECTED_COUNT or sum(row["bytes"] for row in ipn_rows) != IPN_EXPECTED_BYTES:
        raise ValueError("IPN frozen total changed")
    if len({row["file_id"] for row in ipn_rows}) != len(ipn_rows):
        raise ValueError("IPN file ids are not globally unique")
    csafe = select_csafe(_article(CSAFE["article_id"]))
    detailed = {
        "schema_version": 1,
        "experiment": "E32/C4-R1b-corrective-acquisition",
        "state": "selection_frozen_no_selected_bytes_claimed",
        "label_invariant": {"real": 0, "ai": 1},
        "ipn": {
            "umbrella": IPN_UMBRELLA,
            "license": "CC BY 4.0",
            "role": "development",
            "device_count": len(IPN_ARTICLES),
            "natural_count": len(ipn_rows),
            "natural_bytes": sum(row["bytes"] for row in ipn_rows),
            "assets": ipn_rows,
        },
        "csafe": csafe,
        "boundaries": [
            "IPN-NFID is DEVELOPMENT and cannot fit data, model, representation, threshold or policy.",
            "CSAFE iPhone14 is role-free until archive inventory, natural-only selection and audit pass.",
            "The owner gallery is absent.",
            "No selected image byte was downloaded by metadata freeze.",
        ],
    }
    raw = _json_bytes(detailed)
    _write_atomic(SELECTION, detailed)
    compact = {
        "schema_version": 1,
        "experiment": detailed["experiment"],
        "state": detailed["state"],
        "detailed_selection_sha256": _sha256(raw),
        "detailed_selection_bytes": len(raw),
        "ipn_natural_count": len(ipn_rows),
        "ipn_natural_bytes": sum(row["bytes"] for row in ipn_rows),
        "ipn_device_count": len(IPN_ARTICLES),
        "csafe_archive": {key: csafe[key] for key in ("file_id", "name", "bytes", "md5")},
        "new_selected_bytes_downloaded_by_freeze": 0,
    }
    _write_atomic(EVIDENCE, compact)
    return compact


def _selection() -> dict[str, Any]:
    payload = json.loads(SELECTION.read_text())
    if payload.get("state") != "selection_frozen_no_selected_bytes_claimed":
        raise ValueError("R1b acquisition selection is not frozen")
    return payload


def _ensure_capacity(remaining: int) -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(OUTPUT_ROOT).free
    if free < remaining + MIN_FREE_BYTES:
        raise OSError(f"insufficient free space: need {remaining + MIN_FREE_BYTES:,}, have {free:,}")


def _download(asset: Mapping[str, Any], destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected_bytes = int(asset["bytes"])
    expected_md5 = str(asset["md5"])
    if destination.exists():
        if destination.stat().st_size != expected_bytes or _md5(destination) != expected_md5:
            raise ValueError(f"completed file contract mismatch: {destination}")
        return {"state": "already_complete", "bytes": expected_bytes, "md5": expected_md5}
    partial = destination.with_suffix(destination.suffix + ".partial")
    current = partial.stat().st_size if partial.exists() else 0
    if current > expected_bytes:
        raise ValueError(f"partial exceeds expected size: {partial}")
    _ensure_capacity(expected_bytes - current)
    command = [
        "/usr/bin/curl", "--fail", "--location", "--silent", "--show-error",
        "--connect-timeout", "30", "--retry", "8", "--retry-delay", "5",
        "--retry-all-errors", "--speed-limit", "1024", "--speed-time", "120",
        "--output", str(partial),
    ]
    if current:
        command.extend(["--continue-at", "-"])
    command.append(str(asset["url"]))
    subprocess.run(command, check=True)
    if partial.stat().st_size != expected_bytes:
        raise ValueError(f"download size mismatch: {destination}")
    digest = _md5(partial)
    if digest != expected_md5:
        raise ValueError(f"download MD5 mismatch: {destination}")
    partial.replace(destination)
    return {"state": "downloaded", "bytes": expected_bytes, "md5": digest}


def download_ipn(workers: int = 4) -> dict[str, Any]:
    payload = _selection()
    assets = payload["ipn"]["assets"]

    def fetch(asset: Mapping[str, Any]) -> dict[str, Any]:
        destination = OUTPUT_ROOT / "real" / "ipn_nfid" / "natural" / asset["device"] / f"{asset['file_id']}.jpg"
        result = _download(asset, destination)
        return {**result, "file_id": asset["file_id"], "device": asset["device"], "path": str(destination.relative_to(OUTPUT_ROOT))}

    rows = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch, asset) for asset in assets]
        for index, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if index % 40 == 0 or index == len(futures):
                print(f"IPN {index}/{len(futures)} complete", flush=True)
    rows.sort(key=lambda row: row["file_id"])
    receipt = {
        "schema_version": 1,
        "state": "ipn_download_complete_md5_verified",
        "selection_sha256": _sha256(SELECTION.read_bytes()),
        "files": len(rows),
        "bytes": sum(row["bytes"] for row in rows),
        "devices": len({row["device"] for row in rows}),
        "rows": rows,
    }
    _write_atomic(IPN_RECEIPT, receipt)
    return {key: receipt[key] for key in ("state", "files", "bytes", "devices", "selection_sha256")}


def download_csafe() -> dict[str, Any]:
    payload = _selection()
    asset = payload["csafe"]
    destination = OUTPUT_ROOT / "real" / "csafe" / "archives" / str(asset["name"])
    result = _download(asset, destination)
    receipt = {
        "schema_version": 1,
        "state": "csafe_iphone14_download_complete_md5_verified",
        "selection_sha256": _sha256(SELECTION.read_bytes()),
        "path": str(destination.relative_to(OUTPUT_ROOT)),
        **result,
    }
    _write_atomic(CSAFE_RECEIPT, receipt)
    return receipt


def range_plan(start: int, stop: int, workers: int = CSAFE_RANGE_WORKERS) -> list[tuple[int, int]]:
    if start < 0 or stop <= start or workers <= 0:
        raise ValueError("invalid byte-range plan")
    width = ((stop - start) + workers - 1) // workers
    return [(left, min(stop - 1, left + width - 1)) for left in range(start, stop, width)]


def parse_content_range(value: str | None) -> tuple[int, int, int]:
    match = CONTENT_RANGE_RE.fullmatch(value or "")
    if match is None:
        raise ValueError(f"invalid Content-Range: {value!r}")
    return tuple(int(part) for part in match.groups())


def _download_range(url: str, path: Path, start: int, end: int, total: int) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = end - start + 1
    last_error: Exception | None = None
    for delay in RETRY_DELAYS:
        current = path.stat().st_size if path.exists() else 0
        if current == expected:
            return {"path": path, "bytes": current, "state": "already_complete"}
        if current > expected:
            raise ValueError(f"range partial exceeds expected size: {path}")
        if delay:
            time.sleep(delay)
        requested = start + current
        try:
            with requests.get(
                url,
                headers={"Range": f"bytes={requested}-{end}"},
                stream=True,
                timeout=(20, 120),
                allow_redirects=True,
            ) as response:
                response.raise_for_status()
                if response.status_code != 206:
                    raise ValueError(f"range request returned HTTP {response.status_code}")
                found = parse_content_range(response.headers.get("Content-Range"))
                if found != (requested, end, total):
                    raise ValueError(
                        f"range response mismatch: expected {(requested, end, total)}, found {found}"
                    )
                with path.open("ab") as handle:
                    for chunk in response.iter_content(8 * 1024**2):
                        if chunk:
                            handle.write(chunk)
        except (requests.RequestException, OSError) as error:
            last_error = error
            continue
    found_bytes = path.stat().st_size if path.exists() else 0
    if found_bytes != expected:
        raise RuntimeError(f"range failed after retries: {start}-{end}") from last_error
    return {"path": path, "bytes": found_bytes, "state": "downloaded"}


def _assemble_ranges(
    prefix: Path,
    ranges: list[Path],
    destination: Path,
    expected_bytes: int,
    expected_md5: str,
) -> dict[str, Any]:
    assembled = destination.with_suffix(destination.suffix + ".parallel.partial")
    if assembled.exists():
        raise ValueError(f"stale assembled partial requires review: {assembled}")
    digest = hashlib.md5(usedforsecurity=False)
    written = 0
    with assembled.open("xb") as target:
        for source in (prefix, *ranges):
            with source.open("rb") as handle:
                while chunk := handle.read(8 * 1024**2):
                    target.write(chunk)
                    digest.update(chunk)
                    written += len(chunk)
    if written != expected_bytes:
        raise ValueError(f"assembled byte mismatch: expected {expected_bytes}, found {written}")
    found_md5 = digest.hexdigest()
    if found_md5 != expected_md5:
        raise ValueError(f"assembled MD5 mismatch: expected {expected_md5}, found {found_md5}")
    assembled.replace(destination)
    return {"bytes": written, "md5": found_md5}


def download_csafe_ranges() -> dict[str, Any]:
    payload = _selection()
    asset = payload["csafe"]
    destination = OUTPUT_ROOT / "real" / "csafe" / "archives" / str(asset["name"])
    if destination.exists():
        return download_csafe()
    prefix = destination.with_suffix(destination.suffix + ".partial")
    expected = int(asset["bytes"])
    if not prefix.is_file() or not 0 < prefix.stat().st_size < expected:
        raise ValueError("range recovery requires a non-empty strict-prefix partial")
    prefix_bytes = prefix.stat().st_size
    plan = range_plan(prefix_bytes, expected)
    _ensure_capacity((expected - prefix_bytes) + expected)

    def fetch(bounds: tuple[int, int]) -> dict[str, Any]:
        start, end = bounds
        part = destination.with_name(f"{destination.name}.range-{start}-{end}.partial")
        result = _download_range(str(asset["url"]), part, start, end, expected)
        print(f"CSAFE iPhone14 range {start}-{end} complete", flush=True)
        return result

    with ThreadPoolExecutor(max_workers=CSAFE_RANGE_WORKERS) as executor:
        results = list(executor.map(fetch, plan))
    range_paths = [Path(result["path"]) for result in results]
    assembled = _assemble_ranges(prefix, range_paths, destination, expected, str(asset["md5"]))
    prefix.unlink()
    for path in range_paths:
        path.unlink()
    receipt = {
        "schema_version": 1,
        "state": "csafe_iphone14_download_complete_md5_verified",
        "selection_sha256": _sha256(SELECTION.read_bytes()),
        "path": str(destination.relative_to(OUTPUT_ROOT)),
        "bytes": assembled["bytes"],
        "md5": assembled["md5"],
        "preserved_prefix_bytes": prefix_bytes,
        "range_count": len(plan),
    }
    _write_atomic(CSAFE_RECEIPT, receipt)
    print("CSAFE iPhone14 ranges assembled and whole-file MD5 verified", flush=True)
    return receipt


def status() -> dict[str, Any]:
    return {
        "selection_exists": SELECTION.exists(),
        "ipn_receipt_exists": IPN_RECEIPT.exists(),
        "csafe_receipt_exists": CSAFE_RECEIPT.exists(),
        "free_bytes": shutil.disk_usage(OUTPUT_ROOT).free,
        "partials": [str(path.relative_to(OUTPUT_ROOT)) for path in OUTPUT_ROOT.rglob("*.partial")],
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("freeze", "download-ipn", "download-csafe", "download-csafe-ranges", "status"),
    )
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.command == "freeze":
        result = freeze()
    elif args.command == "download-ipn":
        result = download_ipn(args.workers)
    elif args.command == "download-csafe":
        result = download_csafe()
    elif args.command == "download-csafe-ranges":
        result = download_csafe_ranges()
    else:
        result = status()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
