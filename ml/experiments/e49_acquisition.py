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
OPEN_COMPONENTS_CONTRACT = ROOT / "open_components_contract.json"
OPEN_COMPONENTS_EVIDENCE = ML_ROOT.parent / "evidence" / "e49_open_components_contract.json"
COMMONS_METADATA_CACHE = ROOT / "commons_metadata"

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
AIGC_EXPECTED_SHARDS = 60
AIGC_EXPECTED_ROWS = 125_026
AIGC_EXPECTED_GENERATOR_ROWS = 1_997
AIGC_LOCAL_ROOT = DATA_ROOT / "TheKernel01__AIGC-Detection-Benchmark"
AIGC_LOCAL_REF = DATA_ROOT / ".hf_cache" / "hub" / "datasets--TheKernel01--AIGC-Detection-Benchmark" / "refs" / "main"

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
COMMONS_TARGET_PER_DEVICE = 100
COMMONS_RESERVE_PER_DEVICE = 110
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
    "Category:Taken with Nikon Z 8",
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


def read_aigc_coordinates(files: Sequence[Path]) -> tuple[list[tuple[str, int, int, int]], int]:
    """Read only label/generator metadata from local Parquet shards."""
    import pyarrow.parquet as pq

    coordinates: list[tuple[str, int, int, int]] = []
    total_rows = 0
    for path in files:
        table = pq.read_table(path, columns=["label", "generator"])
        labels = table.column("label").to_pylist()
        generators = table.column("generator").to_pylist()
        if len(labels) != len(generators):
            raise ValueError(f"E49 AIGC metadata columns changed: {path.name}")
        coordinates.extend(
            (path.name, row_index, int(label), int(generator))
            for row_index, (label, generator) in enumerate(zip(labels, generators, strict=True))
        )
        total_rows += len(labels)
    return coordinates, total_rows


def probe_local_aigc(
    root: Path = AIGC_LOCAL_ROOT, revision_ref: Path = AIGC_LOCAL_REF,
) -> dict[str, Any]:
    """Validate the pinned local source and freeze a model-blind StyleGAN2 reserve."""
    files = sorted(path for path in (root / "data").glob("test-*.parquet")
                   if not path.name.startswith("._"))
    expected_names = [f"test-{index:05d}-of-00060.parquet" for index in range(AIGC_EXPECTED_SHARDS)]
    if [path.name for path in files] != expected_names:
        raise ValueError(f"E49 AIGC shard inventory changed: {len(files)}/{AIGC_EXPECTED_SHARDS}")
    if revision_ref.read_text().strip() != AIGC_REVISION:
        raise ValueError("E49 local AIGC revision changed")
    coordinates, total_rows = read_aigc_coordinates(files)
    generator_rows = sum(label == 1 and generator == AIGC_GENERATOR_CODE
                         for _, _, label, generator in coordinates)
    if total_rows != AIGC_EXPECTED_ROWS or generator_rows != AIGC_EXPECTED_GENERATOR_ROWS:
        raise ValueError(
            f"E49 local AIGC population changed: rows={total_rows}, StyleGAN2={generator_rows}"
        )
    reserve = select_aigc_coordinates(coordinates)
    identity_raw = json.dumps(
        [{"identity": row["identity"], "rank": row["rank"]} for row in reserve],
        sort_keys=True, separators=(",", ":"),
    ).encode()
    return {
        "schema_version": 1,
        "state": "e49_local_aigc_metadata_validated_unscored",
        "revision": AIGC_REVISION,
        "shards": len(files),
        "rows": total_rows,
        "stylegan2_rows": generator_rows,
        "reserve_rows": len(reserve),
        "reserve_identity_sha256": hashlib.sha256(identity_raw).hexdigest(),
        "image_column_read": False,
        "model_loaded": False,
    }


def _commons_cache_path(category: str) -> Path:
    slug = category.removeprefix("Category:Taken with ").lower().replace(" ", "-")
    return COMMONS_METADATA_CACHE / f"{slug}.json"


def cached_commons_category(category: str) -> list[dict[str, Any]]:
    """Persist eligible API metadata so interrupted scans resume without moving identities."""
    path = _commons_cache_path(category)
    if path.is_file():
        payload = json.loads(path.read_text())
        if payload.get("state") != "e49_commons_eligible_metadata" or payload.get("category") != category:
            raise ValueError(f"E49 Commons metadata cache changed: {category}")
        rows = payload.get("rows", [])
        if not isinstance(rows, list):
            raise ValueError(f"E49 Commons metadata rows changed: {category}")
        return rows
    rows = fetch_commons_category(category)
    _write_atomic(path, {"schema_version": 1, "state": "e49_commons_eligible_metadata",
                         "category": category, "rows": rows, "model_scores_created": 0})
    return rows


def build_open_components_payload(
    commons_by_category: Mapping[str, Sequence[Mapping[str, Any]]],
    aigc_reserve: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    commons = []
    for category in COMMONS_CATEGORIES:
        if category not in commons_by_category:
            raise ValueError(f"E49 Commons category missing: {category}")
        commons.extend(select_capped(
            commons_by_category[category], COMMONS_RESERVE_PER_DEVICE,
            group_key="uploader", max_per_group=COMMONS_MAX_PER_UPLOADER,
        ))
    if len(commons) != len(COMMONS_CATEGORIES) * COMMONS_RESERVE_PER_DEVICE:
        raise ValueError("E49 Commons reserve count changed")
    if len(aigc_reserve) != AIGC_RESERVE:
        raise ValueError("E49 local AIGC reserve count changed")
    network_bytes = sum(int(row["bytes"]) for row in commons)
    if network_bytes > MAX_NETWORK_BYTES:
        raise ValueError(f"E49 open component network stop exceeded: {network_bytes}")
    return {
        "schema_version": 1,
        "state": "e49_open_components_frozen_untransferred_unscored",
        "namespace": NAMESPACE,
        "role": "FINAL_RESERVE_PENDING_DATAPOINT_COMPONENT",
        "commons": {
            "target_rows": COMMONS_TARGET_PER_DEVICE * len(COMMONS_CATEGORIES),
            "reserve_rows": len(commons), "reserve_per_device": COMMONS_RESERVE_PER_DEVICE,
            "max_per_uploader": COMMONS_MAX_PER_UPLOADER,
            "network_bytes": network_bytes, "rows": commons,
        },
        "aigc": {"repo": AIGC_REPO, "revision": AIGC_REVISION,
                 "target_rows": AIGC_TARGET, "reserve_rows": len(aigc_reserve),
                 "rows": list(aigc_reserve)},
        "missing_component": {"repo": DATAPOINT_REPO, "revision": DATAPOINT_REVISION,
                              "target_rows": DATAPOINT_TARGET_PER_MODEL * len(DATAPOINT_MODELS),
                              "state": "awaiting_author_review"},
        "forbidden": ["image transfer before this contract", "model access", "training",
                      "score-dependent replacement", "claiming complete E49 before Datapoint"],
        "new_image_bytes_downloaded": 0,
        "model_scores_created": 0,
    }


def bind_open_components() -> dict[str, Any]:
    """Freeze the ungated E49 components while Datapoint remains unavailable."""
    if OPEN_COMPONENTS_CONTRACT.exists() or OPEN_COMPONENTS_EVIDENCE.exists():
        raise FileExistsError("E49 open-components contract already exists")
    commons_by_category = {category: cached_commons_category(category)
                           for category in COMMONS_CATEGORIES}
    files = sorted(path for path in (AIGC_LOCAL_ROOT / "data").glob("test-*.parquet")
                   if not path.name.startswith("._"))
    coordinates, total_rows = read_aigc_coordinates(files)
    if (len(files) != AIGC_EXPECTED_SHARDS or total_rows != AIGC_EXPECTED_ROWS
            or AIGC_LOCAL_REF.read_text().strip() != AIGC_REVISION):
        raise ValueError("E49 local AIGC identity changed before open-component bind")
    generator_rows = sum(label == 1 and generator == AIGC_GENERATOR_CODE
                         for _, _, label, generator in coordinates)
    if generator_rows != AIGC_EXPECTED_GENERATOR_ROWS:
        raise ValueError("E49 local StyleGAN2 population changed")
    payload = build_open_components_payload(commons_by_category, select_aigc_coordinates(coordinates))
    raw = _write_atomic(OPEN_COMPONENTS_CONTRACT, payload)
    commons_rows_selected = payload["commons"]["rows"]
    aigc_rows_selected = payload["aigc"]["rows"]
    identity_raw = json.dumps(
        [{"identity": row["identity"], "rank": row["rank"]}
         for row in commons_rows_selected + aigc_rows_selected],
        sort_keys=True, separators=(",", ":"),
    ).encode()
    evidence = {
        "schema_version": 1, "state": payload["state"], "role": payload["role"],
        "commons": {key: value for key, value in payload["commons"].items() if key != "rows"},
        "aigc": {key: value for key, value in payload["aigc"].items() if key != "rows"},
        "missing_component": payload["missing_component"],
        "contract_bytes": len(raw), "contract_sha256": hashlib.sha256(raw).hexdigest(),
        "reserve_identity_sha256": hashlib.sha256(identity_raw).hexdigest(),
        "new_image_bytes_downloaded": 0, "model_scores_created": 0,
    }
    _write_atomic(OPEN_COMPONENTS_EVIDENCE, evidence)
    return evidence


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
            "iiextmetadatafilter": "LicenseShortName|LicenseUrl",
            "iiextmetadatalanguage": "en",
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
    parser.add_argument("command", choices=("probe", "probe-local-aigc", "bind-open-components"))
    args = parser.parse_args()
    if args.command == "probe":
        result = probe()
    elif args.command == "probe-local-aigc":
        result = probe_local_aigc()
    else:
        result = bind_open_components()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
