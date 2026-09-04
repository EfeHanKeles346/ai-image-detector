"""Freeze E49 source metadata before downloading any final-test image."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence

from huggingface_hub import HfApi, hf_hub_download
import requests

from pixelproof.project_paths import DATA_ROOT, ML_ROOT


NAMESPACE = "E49_COMPREHENSIVE_FINAL_V1"
ROOT = DATA_ROOT / "e49"
CONTRACT = ROOT / "source_contract.json"
EVIDENCE = ML_ROOT.parent / "evidence" / "e49_source_contract.json"

DATAPOINT_REPO = "datapointai/text-2-image-human-preferences-2m"
DATAPOINT_REVISION = "e1d8719a2d521eac6c62ee84f329afc2c03ec928"
DATAPOINT_MODELS = (
    "GPT Image 2",
    "Nano Banana 2",
    "Seedream 5.0 Pro",
    "FLUX 2",
    "Ideogram 4.0",
)
DATAPOINT_TARGET_PER_MODEL = 160

AIGC_REPO = "TheKernel01/AIGC-Detection-Benchmark"
AIGC_REVISION = "c91d9024a5a77ef06e2ec681b53f9caf08675663"
AIGC_GENERATOR_CODE = 14
AIGC_GENERATOR = "StyleGAN2"
AIGC_TARGET = 200
AIGC_RESERVE = 240

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
COMMONS_TARGET_PER_DEVICE = 100
COMMONS_RESERVE_PER_DEVICE = 120
COMMONS_MAX_PER_UPLOADER = 20
COMMONS_CATEGORIES = (
    "Category:Taken with iPhone 15 Pro",
    "Category:Taken with iPhone 15 Pro Max",
    "Category:Taken with Google Pixel 8 Pro",
    "Category:Taken with Samsung Galaxy S23 Ultra",
    "Category:Taken with iPhone 14 Pro",
    "Category:Taken with Google Pixel 7 Pro",
    "Category:Taken with iPhone 13 Pro",
    "Category:Taken with Sony ILCE-7M4",
    "Category:Taken with Canon EOS R5",
    "Category:Taken with Fujifilm X-T5",
)
ALLOWED_LICENSE_PREFIXES = ("CC BY", "CC0", "Public domain")
MAX_COMMONS_FILE_BYTES = 8 * 1024**2
MAX_NETWORK_BYTES = 4 * 1024**3


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


def validate_hf_identity(
    payload: Mapping[str, Any], *, repo: str, revision: str, license_id: str
) -> None:
    if payload.get("id") != repo or payload.get("sha") != revision:
        raise ValueError(f"E49 repository identity changed: {repo}")
    card = payload.get("cardData") or {}
    if str(card.get("license", "")).lower() != license_id.lower():
        raise ValueError(f"E49 repository licence changed: {repo}")


def select_capped(
    rows: Sequence[Mapping[str, Any]], count: int, *, group_key: str, max_per_group: int
) -> list[dict[str, Any]]:
    """Select by frozen rank while preventing a repeated contributor from dominating."""
    selected: list[dict[str, Any]] = []
    groups: Counter[str] = Counter()
    for candidate in sorted(rows, key=lambda row: (str(row["rank"]), str(row["identity"]))):
        group = str(candidate[group_key])
        if groups[group] >= max_per_group:
            continue
        selected.append(dict(candidate))
        groups[group] += 1
        if len(selected) == count:
            return selected
    raise ValueError(f"cannot fill E49 capped selection: {len(selected)}/{count}")


def commons_rows(category: str, payloads: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Validate model-blind Commons API metadata and return eligible JPEG originals."""
    output: list[dict[str, Any]] = []
    for payload in payloads:
        pages = payload.get("query", {}).get("pages", {}) or {}
        page_rows = pages.values() if isinstance(pages, Mapping) else pages
        for page in page_rows:
            info_rows = page.get("imageinfo") or []
            if len(info_rows) != 1:
                continue
            info = info_rows[0]
            ext = info.get("extmetadata") or {}
            license_name = str((ext.get("LicenseShortName") or {}).get("value", ""))
            url = str(info.get("url", ""))
            size = int(info.get("size", -1))
            width, height = int(info.get("width", -1)), int(info.get("height", -1))
            sha1 = str(info.get("sha1", ""))
            if info.get("mime") != "image/jpeg" or not url.startswith("https://upload.wikimedia.org/"):
                continue
            if not any(license_name.startswith(prefix) for prefix in ALLOWED_LICENSE_PREFIXES):
                continue
            if not (100_000 <= size <= MAX_COMMONS_FILE_BYTES and min(width, height) >= 1_000):
                continue
            if len(sha1) != 40:
                continue
            identity = f"commons:{page['pageid']}:{sha1}"
            output.append({
                "identity": identity,
                "rank": _rank(category, identity),
                "label": 0,
                "source": category.removeprefix("Category:Taken with "),
                "category": category,
                "pageid": int(page["pageid"]),
                "title": str(page["title"]),
                "revision_timestamp": str(info.get("timestamp", "")),
                "uploader": str(info.get("user", "")),
                "url": url,
                "description_url": str(info.get("descriptionurl", "")),
                "bytes": size,
                "width": width,
                "height": height,
                "mime": "image/jpeg",
                "commons_sha1": sha1,
                "license": license_name,
                "license_url": str((ext.get("LicenseUrl") or {}).get("value", "")),
            })
    unique = {row["identity"]: row for row in output}
    return list(unique.values())


def select_aigc_coordinates(
    rows: Iterable[tuple[str, int, int, int]], reserve: int = AIGC_RESERVE
) -> list[dict[str, Any]]:
    """Select StyleGAN2 coordinates from label/generator metadata, never image bytes."""
    candidates = []
    for shard, row_index, label, generator in rows:
        if label != 1 or generator != AIGC_GENERATOR_CODE:
            continue
        identity = f"{shard}:{row_index}"
        candidates.append({
            "identity": identity,
            "rank": _rank(AIGC_GENERATOR, identity),
            "label": 1,
            "source": AIGC_GENERATOR,
            "generator": AIGC_GENERATOR,
            "generator_code": AIGC_GENERATOR_CODE,
            "shard": shard,
            "row_index": row_index,
        })
    selected = sorted(candidates, key=lambda row: (row["rank"], row["identity"]))[:reserve]
    if len(selected) != reserve:
        raise ValueError(f"E49 AIGC reserve changed: {len(selected)}/{reserve}")
    return selected


def _request_json(session: requests.Session, params: Mapping[str, Any]) -> dict[str, Any]:
    for attempt in range(6):
        response = session.get(COMMONS_API, params=params, timeout=(20, 120))
        if response.status_code != 429:
            response.raise_for_status()
            return response.json()
        time.sleep(min(2 ** attempt, 30))
    raise RuntimeError("Wikimedia Commons rate limit persisted after bounded retries")


def fetch_commons_category(category: str, *, max_pages: int = 4) -> list[dict[str, Any]]:
    session = requests.Session()
    session.headers["User-Agent"] = "PixelProof-Research/1.0 (E49 noncommercial benchmark)"
    payloads = []
    continuation: dict[str, Any] = {}
    for _ in range(max_pages):
        params: dict[str, Any] = {
            "action": "query", "format": "json", "formatversion": 2,
            "generator": "categorymembers", "gcmtitle": category,
            "gcmtype": "file", "gcmlimit": 500, "prop": "imageinfo",
            "iiprop": "url|size|mime|sha1|timestamp|user|extmetadata",
        }
        params.update(continuation)
        payload = _request_json(session, params)
        payloads.append(payload)
        continuation = payload.get("continue") or {}
        if not continuation:
            break
    return commons_rows(category, payloads)


def probe() -> dict[str, Any]:
    """Read source metadata only; refuse to claim that gated image access exists."""
    api = HfApi()
    datapoint = api.dataset_info(DATAPOINT_REPO)
    aigc = api.dataset_info(AIGC_REPO)
    validate_hf_identity(datapoint.__dict__, repo=DATAPOINT_REPO,
                         revision=DATAPOINT_REVISION, license_id="cc-by-4.0")
    validate_hf_identity(aigc.__dict__, repo=AIGC_REPO,
                         revision=AIGC_REVISION, license_id="apache-2.0")
    access = False
    access_error = ""
    try:
        hf_hub_download(DATAPOINT_REPO, "data/models/models.parquet", repo_type="dataset",
                        revision=DATAPOINT_REVISION)
        access = True
    except Exception as error:  # the exact Hub exception varies by authentication state
        access_error = type(error).__name__
    return {
        "schema_version": 1,
        "state": "metadata_probe_only",
        "datapoint_revision": datapoint.sha,
        "datapoint_access": access,
        "datapoint_access_error": access_error,
        "aigc_revision": aigc.sha,
        "commons_categories": list(COMMONS_CATEGORIES),
        "target_parents": 2_000,
        "new_image_bytes_downloaded": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("probe",))
    args = parser.parse_args()
    if args.command == "probe":
        print(json.dumps(probe(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
