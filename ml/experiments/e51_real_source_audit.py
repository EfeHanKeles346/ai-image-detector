"""Audit E51 REAL-source metadata without downloading any image payload."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import re
import time
from types import SimpleNamespace
from typing import Any, Iterable, Sequence

import requests
from kaggle.api.kaggle_api_extended import KaggleApi

from pixelproof.project_paths import DATA_ROOT, ML_ROOT


EVIDENCE = ML_ROOT.parent / "evidence" / "e51_real_source_audit.json"
CACHE = DATA_ROOT / "e51" / "source_audit" / "scmi30_public_inventory.json"
KAGGLE_REF = "goyalpuneet/sci30iitrpr"
ZENODO_RECORD = 17317613
IMAGINE_URL = "https://kisi.pcz.pl/imagine/"
IMAGINE_SCRIPT_URL = "https://kisi.pcz.pl/imagine/download/imagine_download.py"
RAISE_URL = "https://loki.disi.unitn.it/RAISE/download.html"


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def summarize_scmi30(metadata: Any, files: Sequence[Any]) -> dict[str, Any]:
    image_files = [row for row in files if str(row.name).lower().endswith((".jpg", ".jpeg"))]
    branches: Counter[str] = Counter()
    device_ids: Counter[str] = Counter()
    for row in image_files:
        parts = str(row.name).split("/")
        if len(parts) < 4:
            raise ValueError(f"Unexpected SCMI30 path: {row.name}")
        branches[parts[1]] += 1
        device_id = parts[2].split("_", 1)[0]
        device_ids[device_id] += 1
    if len(image_files) != 9_937 or set(branches) != {"Random", "Similar"}:
        raise ValueError("SCMI30 public inventory changed")
    if len(device_ids) != 30:
        raise ValueError("SCMI30 device inventory changed")
    return {
        "reference": KAGGLE_REF,
        "version": int(metadata.current_version_number),
        "last_updated": str(metadata.last_updated),
        "license": str(metadata.license_name),
        "files": len(files),
        "image_files": len(image_files),
        "bytes": sum(int(row.total_bytes or 0) for row in files),
        "image_bytes": sum(int(row.total_bytes or 0) for row in image_files),
        "branches": dict(sorted(branches.items())),
        "device_ids": len(device_ids),
        "decision": "eligible_native_train_cal_only",
        "boundary": (
            "Research/education, non-commercial and no-derivatives terms. Split by whole device; "
            "the same publisher may not supply E51 DEVELOPMENT."
        ),
    }


def _kaggle_inventory() -> tuple[Any, list[Any]]:
    if CACHE.exists():
        cached = json.loads(CACHE.read_text())
        metadata = SimpleNamespace(**cached["metadata"])
        files = [SimpleNamespace(**row) for row in cached["files"]]
        return metadata, files
    api = KaggleApi()
    api.authenticate()
    response = requests.get(
        f"https://www.kaggle.com/api/v1/datasets/view/{KAGGLE_REF}", timeout=30
    )
    response.raise_for_status()
    raw_metadata = response.json()
    metadata = SimpleNamespace(
        current_version_number=raw_metadata["currentVersionNumber"],
        last_updated=raw_metadata["lastUpdated"],
        license_name=raw_metadata["licenseName"],
    )
    token = None
    files: list[Any] = []
    while True:
        page = None
        for attempt in range(6):
            try:
                page = api.dataset_list_files(KAGGLE_REF, page_token=token, page_size=200)
                break
            except requests.HTTPError as error:
                if error.response is None or error.response.status_code != 429 or attempt == 5:
                    raise
                retry_after = error.response.headers.get("Retry-After")
                delay = min(float(retry_after), 30.0) if retry_after else min(5.0 * 2**attempt, 30.0)
                time.sleep(delay)
        if page is None:  # pragma: no cover - guarded by the retry loop
            raise RuntimeError("Kaggle inventory produced no page")
        files.extend(page.files or [])
        token = page.next_page_token
        if not token:
            break
        time.sleep(0.25)
    cached = {
        "metadata": {
            "current_version_number": metadata.current_version_number,
            "last_updated": metadata.last_updated,
            "license_name": metadata.license_name,
        },
        "files": [
            {"name": str(row.name), "total_bytes": int(row.total_bytes or 0)}
            for row in files
        ],
    }
    raw = (json.dumps(cached, separators=(",", ":"), sort_keys=True) + "\n").encode()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    temporary = CACHE.with_suffix(".json.part")
    temporary.write_bytes(raw)
    temporary.replace(CACHE)
    return metadata, files


def _get(url: str) -> bytes:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.content


def audit() -> dict[str, Any]:
    if EVIDENCE.exists():
        raise FileExistsError(EVIDENCE)
    metadata, files = _kaggle_inventory()
    scmi30 = summarize_scmi30(metadata, files)
    if (scmi30["files"] != 9_940 or scmi30["bytes"] != 35_592_872_377
            or scmi30["image_bytes"] != 35_592_810_773):
        raise ValueError("SCMI30 byte inventory changed")

    zenodo_raw = _get(f"https://zenodo.org/api/records/{ZENODO_RECORD}")
    zenodo = json.loads(zenodo_raw)
    zenodo_files = {row["key"]: row for row in zenodo["files"]}
    scimd = {
        "record": ZENODO_RECORD,
        "doi": zenodo["doi"],
        "license": zenodo["metadata"]["license"]["id"],
        "image_claim": 17_000,
        "device_models": 17,
        "resolution": "224x224_uniform_resize",
        "archive_bytes": int(zenodo_files["SCIMD-17.zip"]["size"]),
        "archive_md5": zenodo_files["SCIMD-17.zip"]["checksum"],
        "decision": "eligible_auxiliary_train_only",
        "boundary": "Never use as native-camera CAL, DEVELOPMENT or final evidence.",
    }

    imagine = {
        "page_claim_images": 2_816,
        "page_claim_cameras": 60,
        "page_claim_bytes": 26_000_000_000,
        "explicit_license_found": False,
        "decision": "blocked",
        "boundary": "Obtain explicit usage terms before any payload.",
    }
    try:
        imagine_page = _get(IMAGINE_URL)
        imagine_script = _get(IMAGINE_SCRIPT_URL)
    except requests.exceptions.SSLError:
        imagine["verified_tls"] = False
        imagine["boundary"] = (
            "Python cannot verify the publisher TLS chain; obtain explicit terms and a verified "
            "transport before any payload."
        )
    else:
        script_counts = [
            int(value) for value in re.findall(rb"^c = (\d+)\s*$", imagine_script, re.M)
        ]
        imagine.update({
            "verified_tls": True,
            "page_sha256": _sha(imagine_page),
            "download_script_sha256": _sha(imagine_script),
            "script_blocks": len(script_counts),
            "script_image_count": sum(script_counts),
        })
        if sum(script_counts) != 2_816:
            imagine["boundary"] = (
                "Resolve the page/script count mismatch and obtain explicit terms before any payload."
            )

    raise_page = _get(RAISE_URL)
    raise_source = {
        "page_sha256": _sha(raise_page),
        "images": 8_156,
        "cameras": 3,
        "declared_full_bytes": 350_000_000_000,
        "transport": "camera_native_raw",
        "license": "non-commercial research and educational use",
        "decision": "eligible_but_not_primary",
        "boundary": "Too large and too few cameras for E51's primary route; retain as later native spot-check.",
    }

    result = {
        "schema_version": 1,
        "state": "metadata_only_no_image_payload",
        "sources": {
            "scmi30_iitrpr": scmi30,
            "scimd_17": scimd,
            "imagine": imagine,
            "raise": raise_source,
            "dresden": {
                "decision": "blocked",
                "boundary": "Official image host is unavailable; do not substitute a third-party CC0 claim.",
            },
            "socrates": {
                "images": 9_700,
                "smartphones": 103,
                "decision": "blocked",
                "boundary": "Requires a signed licence agreement and emailed password.",
            },
        },
        "routing": {
            "primary_native_train_cal_candidate": "scmi30_iitrpr",
            "auxiliary_resize_hard_negative_candidate": "scimd_17",
            "development_source": None,
            "download_authorized": False,
            "reason": (
                "No independent, explicit-licence DEVELOPMENT publisher passed this audit. "
                "Bind a fresh source before image transfer."
            ),
        },
    }
    raw = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    temporary = EVIDENCE.with_suffix(".json.part")
    temporary.write_bytes(raw)
    temporary.replace(EVIDENCE)
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("audit",))
    parser.parse_args(argv)
    print(json.dumps(audit(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
