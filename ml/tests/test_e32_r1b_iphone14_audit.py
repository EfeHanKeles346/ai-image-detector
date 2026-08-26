from __future__ import annotations

import hashlib
import json
from pathlib import Path

from experiments.e32_r1b_iphone14_audit import owner_exact_hashes


def test_owner_identity_binds_sorted_supported_files(tmp_path: Path) -> None:
    (tmp_path / "b.jpg").write_bytes(b"b")
    (tmp_path / "a.jpeg").write_bytes(b"a")
    (tmp_path / "ignored.mov").write_bytes(b"movie")
    exact, identity_sha, count = owner_exact_hashes(tmp_path)
    identity = [
        {"name": "a.jpeg", "bytes": 1, "sha256": hashlib.sha256(b"a").hexdigest()},
        {"name": "b.jpg", "bytes": 1, "sha256": hashlib.sha256(b"b").hexdigest()},
    ]
    expected = hashlib.sha256((json.dumps(identity, indent=2, sort_keys=True) + "\n").encode()).hexdigest()
    assert count == 2
    assert exact == {hashlib.sha256(b"a").hexdigest(), hashlib.sha256(b"b").hexdigest()}
    assert identity_sha == expected
