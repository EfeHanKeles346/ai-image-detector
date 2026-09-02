"""Acquire the revision-pinned gated ITW-SM final without exposing it to a model."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
from typing import Any, Iterable, Mapping, Sequence

from huggingface_hub import HfApi, hf_hub_download, snapshot_download
from huggingface_hub.errors import GatedRepoError

from pixelproof.project_paths import DATA_ROOT, ML_ROOT


REPO_ID = "dkarageo/itw-sm"
REVISION = "3060094fb576669927134193de3f517d7e64af86"
EXPECTED_FILES = 10_004
EXPECTED_BYTES = 3_573_691_324
EXPECTED_IMAGES = {"0_real": 5_000, "1_fake": 5_000}
NON_IMAGES = {".gitattributes", "LICENSE", "README.md", "metadata.csv"}
MIN_FREE_BYTES = 100 * 1024**3
ROOT = DATA_ROOT / "e43_itwsm"
SNAPSHOT = ROOT / "repository"
RECEIPT = ROOT / "acquisition_receipt.json"
EVIDENCE = ML_ROOT.parent / "evidence" / "e43_itwsm_acquisition.json"


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _write_atomic(path: Path, value: Any) -> bytes:
    raw = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_bytes(raw)
    temporary.replace(path)
    return raw


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def classify_path(name: str) -> tuple[str, str]:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe ITW-SM path: {name!r}")
    if name in NON_IMAGES:
        return "metadata", "metadata"
    if len(path.parts) != 2 or path.parts[0] not in EXPECTED_IMAGES:
        raise ValueError(f"unexpected ITW-SM path: {name!r}")
    if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise ValueError(f"unexpected ITW-SM image suffix: {name!r}")
    return path.parts[0], "image"


def validate_remote_files(files: Sequence[tuple[str, int]]) -> dict[str, Any]:
    if len(files) != EXPECTED_FILES or len({name for name, _ in files}) != len(files):
        raise ValueError("ITW-SM remote file count or uniqueness changed")
    if sum(size for _, size in files) != EXPECTED_BYTES:
        raise ValueError("ITW-SM remote byte count changed")
    images: Counter[str] = Counter()
    metadata = set()
    for name, size in files:
        if size < 0:
            raise ValueError(f"negative ITW-SM file size: {name}")
        group, kind = classify_path(name)
        if kind == "image":
            images[group] += 1
        else:
            metadata.add(name)
    if dict(images) != EXPECTED_IMAGES or metadata != NON_IMAGES:
        raise ValueError(f"ITW-SM remote layout changed: images={images}, metadata={metadata}")
    return {
        "files": len(files),
        "bytes": sum(size for _, size in files),
        "images_by_class": dict(sorted(images.items())),
        "metadata_files": sorted(metadata),
    }


def validate_local_snapshot(snapshot: Path, expected: Mapping[str, int]) -> None:
    for name, size in expected.items():
        path = snapshot / PurePosixPath(name)
        if not path.is_file() or path.stat().st_size != size:
            raise ValueError(f"ITW-SM local file missing or wrong size: {name}")
    actual = {
        path.relative_to(snapshot).as_posix()
        for path in snapshot.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(snapshot).parts
    }
    if actual != set(expected):
        raise ValueError("ITW-SM local payload differs from pinned remote inventory")


def acquire() -> dict[str, Any]:
    if RECEIPT.exists() or EVIDENCE.exists():
        raise FileExistsError("ITW-SM acquisition receipt already exists; no silent replacement")
    ROOT.mkdir(parents=True, exist_ok=True)
    info = HfApi().dataset_info(REPO_ID, revision=REVISION, files_metadata=True)
    if info.sha != REVISION or info.gated != "manual":
        raise ValueError("ITW-SM repository identity/access contract changed")
    expected = {item.rfilename: int(item.size) for item in info.siblings}
    summary = validate_remote_files(list(expected.items()))
    free_before = shutil.disk_usage(ROOT).free
    if free_before < EXPECTED_BYTES + MIN_FREE_BYTES:
        raise OSError("insufficient free space for ITW-SM plus 100 GiB reserve")

    try:
        hf_hub_download(
            repo_id=REPO_ID,
            repo_type="dataset",
            revision=REVISION,
            filename=".gitattributes",
            local_dir=SNAPSHOT,
        )
    except GatedRepoError as error:
        raise PermissionError(
            "ITW-SM content access is still awaiting manual author review; no image transfer started"
        ) from error

    resolved = Path(snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        revision=REVISION,
        local_dir=SNAPSHOT,
        max_workers=4,
    )).resolve()
    if resolved != SNAPSHOT.resolve():
        raise ValueError("ITW-SM downloader resolved an unexpected destination")
    validate_local_snapshot(SNAPSHOT, expected)
    receipt = {
        "schema_version": 1,
        "state": "itwsm_snapshot_complete_unscored",
        "repo_id": REPO_ID,
        "revision": REVISION,
        "license": "ITW-SM research use; gated; non-commercial; no redistribution",
        "role": "E43_UNTOUCHED_FINAL",
        "path": str(SNAPSHOT),
        "inventory": summary,
        "free_bytes_before": free_before,
        "free_bytes_after": shutil.disk_usage(ROOT).free,
        "model_scores_created": 0,
        "boundary": "Acquisition only; no image was opened by a detector and no threshold may use ITW-SM.",
    }
    raw = _write_atomic(RECEIPT, receipt)
    compact = {**receipt, "detailed_receipt_bytes": len(raw), "detailed_receipt_sha256": _sha256(raw)}
    _write_atomic(EVIDENCE, compact)
    return compact


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("acquire",))
    args = parser.parse_args(argv)
    result = acquire() if args.command == "acquire" else None
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
