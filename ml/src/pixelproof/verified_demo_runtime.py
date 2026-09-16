"""Pin the verification manifest itself before using the historical E92 loader.

The old loader and its experimental receipts stay immutable. An intentional runtime
manifest migration requires review and a new pin here, not a self-declared identity.
"""
import hashlib
from pathlib import Path
from pixelproof import e92_demo

MANIFEST_SHA256 = '6bf3e29c8c4bada93d975513f31e8acf8b615d6076b52a53239ccbe06d48a553'
VERIFICATION_ID = 'e92-manifest-pinned-v1'


def verify_manifest(path: Path | None = None) -> None:
    path = e92_demo.MANIFEST if path is None else path
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA256:
        raise ValueError('E92 runtime manifest identity mismatch')


class VerifiedE92Engine(e92_demo.E92Engine):
    def __init__(self):
        verify_manifest()
        super().__init__()
