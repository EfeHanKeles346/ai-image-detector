"""Model-blind, restartable E51 byte and cross-role identity audit.

This is an admission diagnostic, not a training command or a model-quality result.
Frozen manifests are never edited. File names are not used to infer labels.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
from io import BytesIO
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageOps

from experiments.e51_train_cal_realize import _phash_image, _write_atomic
from pixelproof.project_paths import DATA_ROOT, ML_ROOT

ROOT = DATA_ROOT / "e51"
INPUTS = {
    "TRAIN": (ROOT / "manifests/train_parents_unscored.json",
              "41444640273703fb96a9e3e269132ddc21e3056cbb8626e1aa93ea3a64cc77ef"),
    "CAL": (ROOT / "manifests/cal_paired_unscored.json",
            "606882913b0a2193422a8e838a142cff532bc1fe63088c6802eb3c9a43472356"),
}
VERSION = "e51-prefit-rgb-lanczos-v1"
REPORT = ROOT / "audit/prefit_identity_v1.json"
EVIDENCE = ML_ROOT.parent / "evidence/e51_prefit_identity.json"
CACHE = ROOT / "audit/fingerprints_v1"


def fingerprint(raw: bytes) -> dict[str, Any]:
    """Use a single explicit convention; ties are zero, not inverse-complemented."""
    with Image.open(BytesIO(raw)) as image:
        if image.width * image.height > 100_000_000:
            raise ValueError("image exceeds pre-decode pixel budget")
        image.verify()
    with Image.open(BytesIO(raw)) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        grey = np.asarray(rgb.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
        value = 0
        for bit in (grey[:, 1:] > grey[:, :-1]).ravel():
            value = (value << 1) | int(bit)
        pixels = hashlib.sha256(f"RGB:{rgb.width}:{rgb.height}:".encode())
        pixels.update(rgb.tobytes())
        return {
            "version": VERSION,
            "sha256": hashlib.sha256(raw).hexdigest(),
            "bytes": len(raw), "width": rgb.width, "height": rgb.height,
            "pixel_sha256": pixels.hexdigest(),
            "canonical_dhash": f"{value:016x}",
            "phash63": f"{_phash_image(rgb):016x}",
        }


def verify_row(row: Mapping[str, Any], cache: Path) -> dict[str, Any]:
    # Re-read/hash bytes even on resume: an old cache cannot hide a replaced payload.
    raw = Path(str(row["path"])).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != row["sha256"] or len(raw) != int(row["bytes"]):
        raise ValueError(f"payload identity changed: {row['parent_id']}")
    entry = cache / f"{sha}.json"
    facts = json.loads(entry.read_text()) if entry.is_file() else None
    if facts is None:
        facts = fingerprint(raw)
        _write_atomic(entry, facts)
    if facts.get("version") != VERSION or facts.get("sha256") != sha:
        raise ValueError(f"fingerprint cache identity changed: {row['parent_id']}")
    return {"parent_id": row["parent_id"], "role": row["audit_role"],
            "condition": row.get("condition", "original"), "label": row["label"],
            "source": row["source"], "path": row["path"], **facts}


def validate_roles(train: Sequence[Mapping], cal: Sequence[Mapping]) -> None:
    if not train or not cal:
        raise ValueError("empty TRAIN or CAL")
    seen = set()
    for row in train:
        if row["label"] not in (0, 1) or str(row["role"]).upper() != "TRAIN":
            raise ValueError("invalid TRAIN role/label")
        if row["parent_id"] in seen:
            raise ValueError("duplicate TRAIN parent")
        seen.add(row["parent_id"])
    by_parent: dict[str, list[Mapping]] = defaultdict(list)
    for row in cal:
        if row["label"] not in (0, 1) or row["role"] != "CAL":
            raise ValueError("invalid CAL role/label")
        by_parent[row["parent_id"]].append(row)
    if seen & by_parent.keys():
        raise ValueError("TRAIN/CAL parent overlap")
    for rows in by_parent.values():
        if (Counter(r["condition"] for r in rows) != {"original": 1, "q75": 1}
                or len({r["label"] for r in rows}) != 1
                or len({r["source"] for r in rows}) != 1):
            raise ValueError("CAL pairing or label/source conflict")


def cross_role_matches(train: Sequence[Mapping], cal: Sequence[Mapping]) -> list[dict]:
    """Vectorized radius-4 dHash screening, fixed pHash confirmation; not exhaustive similarity."""
    right_hashes = np.array([int(r["canonical_dhash"], 16) for r in cal], dtype=np.uint64)
    matches = {}
    exact_indexes = {}
    for field in ("sha256", "pixel_sha256"):
        index: dict[str, list[int]] = defaultdict(list)
        for j, row in enumerate(cal):
            index[row[field]].append(j)
        exact_indexes[field] = index
    for left in train:
        distances = np.bitwise_count(right_hashes ^ np.uint64(int(left["canonical_dhash"], 16)))
        candidates = set(np.flatnonzero(distances <= 4).tolist())
        for field, index in exact_indexes.items():
            candidates.update(index.get(left[field], []))
        for j in sorted(candidates):
            right = cal[j]
            exact = [field for field in exact_indexes if left[field] == right[field]]
            phash_distance = (int(left["phash63"], 16) ^ int(right["phash63"], 16)).bit_count()
            if not exact and phash_distance > 4:
                continue
            key = (left["parent_id"], right["parent_id"])
            detail = {"train_parent": key[0], "cal_parent": key[1],
                      "train_label": left["label"], "cal_label": right["label"],
                      "cal_condition": right["condition"], "exact_fields": exact,
                      "dhash_distance": int(distances[j]), "phash63_distance": phash_distance}
            # Preserve every paired condition; do not count two transports as two parents.
            matches.setdefault(key, []).append(detail)
    return [item for key in sorted(matches) for item in matches[key]]


def load_inputs() -> tuple[list[dict], list[dict]]:
    result = []
    for role, (path, expected) in INPUTS.items():
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"frozen {role} manifest changed")
        payload = json.loads(raw)
        if payload.get("model_scores_created") != 0:
            raise ValueError(f"{role} manifest no longer unscored")
        result.append([{**row, "audit_role": role} for row in payload["rows"]])
    validate_roles(*result)
    return result[0], result[1]


def audit(workers: int = 4) -> dict[str, Any]:
    if REPORT.exists() or EVIDENCE.exists():
        raise FileExistsError("pre-fit report already exists; preserve audit history")
    train, cal = load_inputs()
    verified = []
    # Distinct source-byte entries avoid simultaneous cache writes for shared bytes.
    cache_rows = {str(row["sha256"]): row for row in train + cal}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, _ in enumerate(pool.map(lambda row: verify_row(row, CACHE), cache_rows.values()), 1):
            if i % 250 == 0:
                print(json.dumps({"verified_unique_bodies": i, "total": len(cache_rows)}), flush=True)
    # Verify every path, not merely one representative for each expected digest.
    with ThreadPoolExecutor(max_workers=workers) as pool:
        verified = list(pool.map(lambda row: verify_row(row, CACHE), train + cal))
    left, right = verified[:len(train)], verified[len(train):]
    matches = cross_role_matches(left, right)
    full = {
        "schema_version": 1, "version": VERSION,
        "state": "cross_role_overlap_requires_review" if matches else "local_cross_role_identity_check_passed",
        "input_manifest_sha256": {role: sha for role, (_, sha) in INPUTS.items()},
        "verified_observations": len(verified), "train_parents": len(train),
        "cal_parents": len({r['parent_id'] for r in cal}),
        "label_counts": {role: dict(Counter(str(r['label']) for r in rows))
                         for role, rows in (("TRAIN", train), ("CAL_observations", cal))},
        "cross_role_matched_parent_pairs": len({(r['train_parent'], r['cal_parent']) for r in matches}),
        "matches": matches,
        "all_payload_bytes_verified": True,
        "model_scores_created": 0, "training_authorized": False,
        "pending_admission": [
            "Review/remove any confirmed cross-role overlap by a score-blind manifest amendment",
            "Complete protected-role and unused E49 reserve identity/perceptual audit",
            "Bind training feature/candidate contract before fit",
        ],
        "limits": [
            "dHash radius 4 plus pHash63 radius 4 is a candidate screen, not proof of no semantic duplicates",
            "No prompt/session grouping is inferred from absent metadata",
            "This report does not certify publisher labels or model performance",
        ],
    }
    raw = _write_atomic(REPORT, full)
    evidence = {k: v for k, v in full.items() if k != "matches"}
    evidence.update(report_sha256=hashlib.sha256(raw).hexdigest(), report_path=str(REPORT),
                    match_examples=matches[:12])
    _write_atomic(EVIDENCE, evidence)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    args = parser.parse_args()
    print(json.dumps(audit(args.workers), indent=2))


if __name__ == "__main__":
    main()
