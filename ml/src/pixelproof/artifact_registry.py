"""Offline verification for every model artifact admitted to the runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(repo_root: Path = DEFAULT_REPO_ROOT) -> dict[str, Any]:
    path = repo_root / "artifacts.manifest.json"
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise RuntimeError(f"artifact manifest missing: {path}") from None
    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("artifacts"), list):
        raise RuntimeError(f"unsupported artifact manifest: {path}")
    return manifest


def _verify_hash(path: Path, expected: str, artifact_id: str, issues: list[str]) -> None:
    if not path.is_file():
        issues.append(f"{artifact_id}: missing {path}")
        return
    actual = sha256_file(path)
    if actual != expected:
        issues.append(f"{artifact_id}: SHA-256 mismatch for {path} (got {actual})")


def _verify_entry(entry: dict[str, Any], repo_root: Path, issues: list[str]) -> None:
    artifact_id = entry["id"]
    kind = entry["kind"]
    if kind == "file":
        _verify_hash(repo_root / entry["path"], entry["sha256"], artifact_id, issues)
        return

    if kind == "huggingface":
        try:
            from huggingface_hub import snapshot_download

            snapshot = Path(snapshot_download(
                entry["repo_id"],
                revision=entry["revision"],
                local_files_only=True,
            ))
        except Exception as error:
            issues.append(
                f"{artifact_id}: pinned Hugging Face snapshot unavailable locally "
                f"({type(error).__name__}: {error})"
            )
            return
        for relative, expected in entry["files"].items():
            _verify_hash(snapshot / relative, expected, artifact_id, issues)
        return

    if kind == "checkout":
        checkout = repo_root / entry["path"]
        if not checkout.is_dir():
            issues.append(f"{artifact_id}: optional checkout missing {checkout}")
            return
        revision = subprocess.run(
            ["git", "-C", str(checkout), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if revision != entry["revision"]:
            issues.append(f"{artifact_id}: revision mismatch (got {revision or 'unknown'})")
        for relative, expected in entry["files"].items():
            _verify_hash(checkout / relative, expected, artifact_id, issues)
        return

    issues.append(f"{artifact_id}: unknown artifact kind {kind!r}")


def verify_registry(
    repo_root: Path = DEFAULT_REPO_ROOT,
    groups: set[str] | None = None,
    include_optional: bool = False,
) -> dict[str, Any]:
    manifest = load_manifest(repo_root)
    issues: list[str] = []
    checked: list[str] = []
    for entry in manifest["artifacts"]:
        if groups is not None and entry.get("group") not in groups:
            continue
        if entry.get("optional") and not include_optional:
            continue
        checked.append(entry["id"])
        _verify_entry(entry, repo_root, issues)
    return {"ok": not issues, "checked": checked, "issues": issues}


def prepare_external_snapshots(repo_root: Path = DEFAULT_REPO_ROOT) -> list[str]:
    """Fetch only redistributable external snapshots; project weights stay owner-supplied."""
    from huggingface_hub import snapshot_download

    prepared = []
    for entry in load_manifest(repo_root)["artifacts"]:
        if entry["kind"] != "huggingface":
            continue
        snapshot_download(entry["repo_id"], revision=entry["revision"])
        prepared.append(entry["id"])
    return prepared


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["check", "prepare"], default="check", nargs="?")
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--group", action="append", dest="groups")
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps({"prepared": prepare_external_snapshots(args.repo_root.resolve())}, indent=2))
    report = verify_registry(
        args.repo_root.resolve(),
        set(args.groups) if args.groups else None,
        args.include_optional,
    )
    print(json.dumps(report, indent=2))
    if not report["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
