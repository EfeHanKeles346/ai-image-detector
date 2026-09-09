"""Resolve protected E51 audit inputs locally; never download or extract images."""

from __future__ import annotations

from collections import Counter
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Mapping
import zipfile

from experiments.e51_protected_inventory import identity_rows
from experiments.e51_train_cal_realize import _write_atomic
from pixelproof.project_paths import DATA_ROOT, ML_ROOT

ROOT = DATA_ROOT / "e51"
INVENTORY = ROOT / "audit/protected_inventory_v1.json"
INVENTORY_EVIDENCE = ML_ROOT.parent / "evidence/e51_protected_inventory.json"
OUTPUT = ROOT / "audit/offline_body_locators_v1.json"
EVIDENCE = ML_ROOT.parent / "evidence/e51_offline_bodies.json"
ARCHIVES = {
    ("e43_dda_coco", "COCO-val2017"): DATA_ROOT / "e43_dda_coco/archives/val2017.zip",
    ("e43_dda_coco", "DDA-COCO"): DATA_ROOT / "e34_dda_coco/archives/DDA-COCO.zip",
    ("e45_mediaeval_itwsm", ""): DATA_ROOT / "e45_mediaeval_itwsm/archives/itw-sm-sid-val.zip",
}


def relative_body_path(manifest: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    if ".." in path.parts:
        raise ValueError("relative body path traverses manifest directory")
    return manifest.parent / path


def safe_zip_info(bundle: zipfile.ZipFile, member: str) -> zipfile.ZipInfo:
    path = PurePosixPath(member)
    if path.is_absolute() or ".." in path.parts or "\\" in member:
        raise ValueError("unsafe ZIP member")
    info = bundle.getinfo(member)
    if info.is_dir() or (info.external_attr >> 16 & 0o170000) == 0o120000:
        raise ValueError("ZIP image is not a regular file")
    if info.flag_bits & 1 or not 0 < info.file_size <= 200 * 1024**2:
        raise ValueError("encrypted or oversized ZIP image")
    return info


def resolve(row: Mapping, manifest: Path, bundles: Mapping[Path, zipfile.ZipFile]) -> dict:
    expected = row.get("bytes", row.get("expected_bytes"))
    if row.get("path"):
        path = relative_body_path(manifest, str(row['path']))
        if not path.is_file():
            raise FileNotFoundError(str(path))
        size = path.stat().st_size
        locator = {"kind": "file", "path": str(path)}
    elif row.get("member"):
        key = (manifest.parent.name, str(row.get("archive", "")))
        if key not in ARCHIVES:
            raise ValueError(f"unbound archive route: {key}")
        archive = ARCHIVES[key]
        info = safe_zip_info(bundles[archive], str(row['member']))
        size = info.file_size
        locator = {"kind": "zip", "archive": str(archive), "member": info.filename,
                   "crc32": f"{info.CRC:08x}"}
    else:
        raise ValueError("metadata-only row has no bound local body locator")
    if expected is not None and size != int(expected):
        raise ValueError("local body size differs from protected metadata")
    locator.update(bytes=size, sha256=row.get('sha256'),
                   identity=str(row.get('record_id') or row.get('identity') or row.get('parent_id')))
    return locator


def bind() -> dict:
    if OUTPUT.exists() or EVIDENCE.exists():
        raise FileExistsError("offline locator inventory already frozen")
    raw = INVENTORY.read_bytes()
    if raw != INVENTORY_EVIDENCE.read_bytes():
        raise ValueError("protected inventory does not match committed evidence")
    inventory = json.loads(raw)
    if inventory.get('model_scores_created') != 0:
        raise ValueError("protected inventory is no longer model blind")
    locators, missing, metadata_only, summaries = {}, [], [], []
    with ExitStack() as stack:
        bundles = {path: stack.enter_context(zipfile.ZipFile(path)) for path in ARCHIVES.values()}
        for source in inventory['sources']:
            manifest = Path(source['path'])
            source_raw = manifest.read_bytes()
            if hashlib.sha256(source_raw).hexdigest() != source['sha256']:
                raise ValueError(f"protected manifest changed: {manifest}")
            counts = Counter()
            for row in identity_rows(json.loads(source_raw)):
                record = {"manifest": str(manifest), "identity": str(
                    row.get('record_id') or row.get('identity') or row.get('parent_id'))}
                if not row.get('path') and not row.get('member'):
                    metadata_only.append(record)
                    counts['metadata_only'] += 1
                    continue
                try:
                    location = resolve(row, manifest, bundles)
                except (OSError, ValueError, KeyError) as error:
                    missing.append({**record, 'error': f'{type(error).__name__}: {error}'})
                    counts['unresolved_body'] += 1
                    continue
                key = json.dumps({k: location.get(k) for k in ('kind', 'path', 'archive', 'member')}, sort_keys=True)
                if key in locators:
                    previous = locators[key]
                    if (previous.get('sha256') and location.get('sha256')
                            and previous['sha256'] != location['sha256']):
                        raise ValueError("conflicting expected SHA-256 for same protected body")
                    if location.get('sha256'):
                        previous['sha256'] = location['sha256']
                    previous['protected_by'].append(record)
                else:
                    locators[key] = {**location, 'protected_by': [record]}
                counts['resolved_body'] += 1
            summaries.append({**source, 'locator_counts': dict(counts)})
    rows = [locators[key] for key in sorted(locators)]
    report = {
        'schema_version': 1, 'state': 'offline_body_inventory_frozen_not_pixel_audited',
        'protected_inventory_sha256': hashlib.sha256(raw).hexdigest(),
        'unique_body_locators': len(rows), 'kind_counts': dict(Counter(r['kind'] for r in rows)),
        'body_bytes_to_verify': sum(r['bytes'] for r in rows),
        'bodies_without_prior_sha256': sum(not r.get('sha256') for r in rows),
        'unresolved_body_rows': len(missing), 'metadata_only_rows': len(metadata_only),
        'sources': summaries, 'rows': rows, 'unresolved': missing, 'metadata_only': metadata_only,
        'image_bytes_downloaded': 0, 'new_image_bodies_decoded': 0, 'model_scores_created': 0,
        'training_authorized': False,
        'next': 'Verify/read these bodies and compare canonical fingerprints; separately close metadata-only reserves. Do not fit on this locator receipt alone.',
    }
    raw = _write_atomic(OUTPUT, report)
    evidence = {k: v for k, v in report.items() if k not in {'rows', 'metadata_only', 'unresolved'}}
    evidence.update(locator_manifest_sha256=hashlib.sha256(raw).hexdigest(),
                    unresolved_examples=missing[:10])
    _write_atomic(EVIDENCE, evidence)
    return evidence


if __name__ == '__main__':
    print(json.dumps(bind(), indent=2))
