"""Offline, repository-backed E92 inference. Frozen experiments are never rewritten.

Run with PYTHONPATH=ml:ml/src and PIXELPROOF_DATA_ROOT pointing to the existing disk.
This is a local research adapter, not a redistributable weight bundle.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

from pixelproof.project_paths import DATA_ROOT, ML_ROOT

AI_CUT = 0.07940196245908739
REAL_CUT = 0.011505939625203613
MODEL_ID = 'E92'
GUARD_ID = 'e92-stability-v1'
CANDIDATE_SHA = '3a68c50d7cabd17d74c90bdcaf3b74aaacbc6c07e0bf28e332b1b91f99c9ef35'
MANIFEST = ML_ROOT.parent / 'evidence/e93_runtime_manifest.json'


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def guarded_outcome(scores, reference) -> tuple[str, str]:
    """Fixed selective output, not calibrated confidence or a learned OOD detector."""
    s, r = np.asarray(scores, dtype=float), np.asarray(reference, dtype=float)
    if s.shape != (2,) or r.shape != (2,) or not np.isfinite([s, r]).all() or \
            np.any(s < 0) or np.any(s > 1) or np.any(r < 0) or np.any(r > 1):
        raise ValueError('Two finite bounded paired scores required')
    if np.all(s >= AI_CUT):
        return 'ai_signal', 'stable_signal'
    if np.all(s < REAL_CUT) and np.all(r < AI_CUT):
        return 'no_clear_signal', 'limited_negative_evidence'
    return 'uncertain', 'inconsistent_or_borderline'


def social_view(image: Image.Image) -> Image.Image:
    """Exact frozen social-Q75 convention, entirely in memory."""
    result = ImageOps.exif_transpose(image).convert('RGB')
    if max(result.size) > 1080:
        scale = 1080 / max(result.size)
        result = result.resize((max(1, round(result.width * scale)),
                                max(1, round(result.height * scale))), Image.Resampling.LANCZOS)
    stream = io.BytesIO()
    result.save(stream, format='JPEG', quality=75, subsampling=2, optimize=False)
    stream.seek(0)
    with Image.open(stream) as decoded:
        return decoded.convert('RGB').copy()


class E92Engine:
    def __init__(self):
        # Set before importing HF/timm. No helper is allowed to acquire a missing asset.
        os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                          HF_HUB_DISABLE_TELEMETRY='1')
        manifest = json.loads(MANIFEST.read_text())
        for root, files in ((DATA_ROOT, manifest['data_files']), (ML_ROOT.parent, manifest['code_files'])):
            for relative, expected in files.items():
                if digest(root / relative) != expected:
                    raise ValueError('E92 runtime component identity mismatch')
        import joblib
        import torch
        import timm
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file
        from timm.models.vision_transformer import checkpoint_filter_fn
        from experiments import e42_features as dino, e78_model as dear
        from experiments.e71_features import load_encoder
        from experiments.e80_development import dear_features
        from experiments.e92_model import predict

        torch.set_num_threads(2)
        self.torch, self.dino, self.predict_head = torch, dino, predict
        self.dear_features = dear_features
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.head = joblib.load(DATA_ROOT / 'e43/e43_small_predev.joblib')['head']
        with np.load(DATA_ROOT / 'e92/correction.npz', allow_pickle=False) as archive:
            if str(archive['reference_sha256']) != manifest['data_files']['e43/e43_small_predev.joblib'] or \
                    str(archive['contract_sha256']) != manifest['data_files']['e92/fit_contract.json']:
                raise ValueError('E92 fitted archive binding mismatch')
            self.arrays = {key: archive[key] for key in archive.files
                           if key not in ('reference_sha256', 'contract_sha256')}
        snapshot = Path(snapshot_download(dino.DINO_REPO_ID, local_files_only=True))
        weights = snapshot / 'model.safetensors'
        if digest(weights) != dino.DINO_WEIGHT_SHA256:
            raise ValueError('DINO cached weight identity mismatch')
        backbone = timm.create_model(dino.DINO_MODEL_ID, pretrained=False, num_classes=0, img_size=224)
        backbone.load_state_dict(checkpoint_filter_fn(load_file(str(weights)), backbone), strict=True)
        config = timm.data.resolve_data_config({}, model=backbone)
        self.backbone = backbone.to(self.device).eval()
        self.mean = torch.tensor(config['mean'], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(config['std'], device=self.device).view(1, 3, 1, 1)
        self.clip, self.clip_preprocess, self.clip_device = load_encoder()
        self.dear = dear.load(DATA_ROOT / 'e78/dear_r.pth', self.device)
        for network in (self.backbone, self.clip, self.dear):
            for parameter in network.parameters():
                parameter.requires_grad_(False)

    def score_views(self, images: list[Image.Image]) -> tuple[np.ndarray, np.ndarray]:
        """Maximum eight image views; fixed CLIP3 and DEAR9 microbatches."""
        if not 1 <= len(images) <= 8:
            raise ValueError('Expected one to eight native image views')
        torch = self.torch
        from threadpoolctl import threadpool_limits
        from experiments.e71_development import clip_aggregate
        packs, clip_vectors = [], []
        with torch.inference_mode(), threadpool_limits(limits=2):
            for image in images:
                pack = self.dino.texture_crops(self.dino.transport_image(image, 'clean'))
                packs.extend(pack)
                clip_input = torch.stack([self.clip_preprocess(Image.fromarray(crop)) for crop in pack])
                raw = self.clip.encode_image(clip_input.to(self.clip_device)).float().cpu().numpy()
                clip_vectors.append(clip_aggregate(raw))
            tensor = torch.from_numpy(np.stack(packs)).to(self.device).permute(0, 3, 1, 2).float().div_(255)
            blocks = self.backbone.forward_intermediates((tensor - self.mean) / self.std,
                indices=list(self.dino.BLOCKS['small']), return_prefix_tokens=True,
                norm=True, intermediates_only=True)
            tokens = torch.stack([block[1][:, 0, :] for block in blocks], dim=1).float().cpu().numpy()
            original = self.dino.aggregate_tokens(tokens, len(images))
            forensic = self.dear_features(self.dear, np.stack(packs).reshape(len(images), 3, 224, 224, 3))
            scores = self.predict_head(self.head, original, np.stack(clip_vectors), forensic, self.arrays)
            reference = self.head.predict_proba(original)[:, 1]
        if not np.isfinite([scores, reference]).all() or np.any(scores < 0) or np.any(scores > 1):
            raise ValueError('Invalid native prediction')
        return scores, reference

    def analyze(self, image: Image.Image) -> tuple[str, str]:
        scores, reference = self.score_views([image, social_view(image)])
        return guarded_outcome(scores, reference)
