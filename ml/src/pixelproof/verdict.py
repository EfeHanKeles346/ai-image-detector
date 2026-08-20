# =============================================================================
# verdict.py — WHAT THIS FILE DOES
# -----------------------------------------------------------------------------
# The decision layer the demo serves: frozen external detectors behind the
# asymmetric abstention band that E22–E24 measured. This is the only place in
# the project that turns a score into a user-facing decision, and every
# constant in it is a measurement with an experiment id next to it.
#
# THE CONTRACT (all measured, see ml/EXPERIMENTS.md)
# -----------------------------------------------------------------------------
# - Input policy (E23b): before B-Free sees an image, its long side is capped
#   at 2048px. Uncapped, 12-megapixel authentic photos read as "generated"
#   (38% FP on the owner's own iPhone in E24).
# - Verdicts (E23a): "ai" or "insufficient" — never "real". At every miss
#   budget at least one authentic pipeline got 0% "real" coverage, and a
#   "real" verdict is the one output through which AI content can be
#   laundered. "No AI evidence" is not a certificate of authenticity.
# - Thresholds (E24): worst-source rule over the 12-pipeline calibration
#   library (Defactify real + 10 forensic sources + the iPhone pipeline).
#   Measured operating points: B-Free+cap 9.7% FP on the iPhone evaluation
#   half at 62.2% macro recall; CF 1.0% FP at 28.0% recall.
# - Compression caveat (E23c): below ~0.3 bytes/pixel the input sits in the
#   degraded regime where recall drops (65→43 for B-Free) — flagged, not
#   silently ignored.
#
# PER BLOCK
# -----------------------------------------------------------------------------
# CommunityForensicsArm  MIT-licensed ViT-S, loaded from the local HF cache.
#                        Always servable.
# BFreeArm               GRIP-UNINA B-Free — informational/nonprofit licence,
#                        so it loads only when the official checkout exists
#                        AND PIXELPROOF_BFREE=1 acknowledges that scope.
# decide()               pure band logic, unit-testable without any model.
# VerdictService         loads whatever arms are available; B-Free primary
#                        (better recall), CF fallback/second signal.
# =============================================================================

from __future__ import annotations

import importlib
import os
import sys
import time
from pathlib import Path

import torch
from PIL import Image

# --- measured constants ------------------------------------------------------
CAP_PX = 2048            # E23b: rescues megapixel pipelines (NIST 35.2% -> 8.8% FP)
CF_T_AI = 0.6617392      # E24: worst-source threshold, 12-pipeline library
BFREE_T_AI = 1.0439179   # E24: same rule, capped B-Free arm
COMPRESSED_BPP = 0.30    # E23c: below this, recall sits in the degraded regime
LICENCE_ENV = "PIXELPROOF_BFREE"


def capped(picture: Image.Image) -> tuple[Image.Image, bool]:
    """E23b input policy: long side <= CAP_PX before scoring."""
    width, height = picture.size
    long_side = max(width, height)
    if long_side <= CAP_PX:
        return picture, False
    scale = CAP_PX / long_side
    return picture.resize(
        (round(width * scale), round(height * scale)), Image.LANCZOS
    ), True


def decide(score: float, threshold: float) -> str:
    """The asymmetric band (E23a): 'ai' above the line, honesty below it."""
    return "ai" if score >= threshold else "insufficient"


def combine(bands: dict[str, str]) -> tuple[str, list[str]]:
    """E26 — the OR rule: any arm above its own worst-source threshold decides.

    Measured on every cached score set (2026-08-20): the arms' false positives
    live on different sources, so the union stays at 9.7% worst-source FP
    (unchanged) while recall rises everywhere the arms disagree — Midjourney
    7%→14%, FLUX 38%→64.5%, and the GPT-family images B-Free is blind to are
    caught by CF. A single primary arm was discarding correct votes.
    """
    triggered = [name for name, band in bands.items() if band == "ai"]
    return ("ai" if triggered else "insufficient"), triggered


class CommunityForensicsArm:
    name = "cf_vit"
    label = "Community-Forensics ViT-S"
    threshold = CF_T_AI

    def __init__(self, device: torch.device) -> None:
        from huggingface_hub import snapshot_download
        from transformers import ViTForImageClassification, ViTImageProcessor

        local = snapshot_download(
            "buildborderless/CommunityForensics-DeepfakeDet-ViT",
            local_files_only=True,   # E21 cached it; serving never downloads
        )
        self.model = (
            ViTForImageClassification.from_pretrained(local, local_files_only=True)
            .to(device).eval()
        )
        self.processor = ViTImageProcessor.from_pretrained(local, local_files_only=True)
        self.device = device

    @torch.inference_mode()
    def score(self, picture: Image.Image) -> float:
        inputs = self.processor(images=picture, return_tensors="pt")
        logits = self.model(pixel_values=inputs["pixel_values"].to(self.device)).logits
        return float(logits[0, 0].cpu())


class BFreeArm:
    name = "bfree"
    label = "B-Free (GRIP-UNINA)"
    threshold = BFREE_T_AI

    def __init__(self, device: torch.device, repo: Path) -> None:
        import yaml
        from torchvision.transforms import Compose

        config_path = repo / "weights/BFREE_dino2reg4/config.yaml"
        config = yaml.safe_load(config_path.read_text())
        weights_path = config_path.parent / config["weights_file"]
        sys.path.insert(0, str(repo / "code"))
        try:
            networks = importlib.import_module("networks")
            normalization = importlib.import_module("utils.normalization")
        finally:
            sys.path.pop(0)
        model = networks.get_network(config["arch"])
        self.model = networks.load_weights(model, weights_path).to(device).eval()
        self.transform = Compose(normalization.get_list_norm(config["norm_type"]))
        self.device = device

    @torch.inference_mode()
    def score(self, picture: Image.Image) -> float:
        tensor = self.transform(picture).unsqueeze(0).to(self.device)
        output = self.model(tensor)
        if output.shape[1] == 1:
            return float(output[0, 0].cpu())
        return float((output[0, 1] - output[0, 0]).cpu())


class VerdictService:
    """Loads whatever arms are available and serves the measured band."""

    def __init__(self, device: torch.device, repo_root: Path) -> None:
        self.arms: list = []
        began = time.perf_counter()
        try:
            self.arms.append(CommunityForensicsArm(device))
        except Exception as error:  # missing cache/transformers: demo degrades gracefully
            print(f"[verdict] CF-ViT yüklenemedi: {error}")
        bfree_repo = repo_root / "external/B-Free"
        if os.environ.get(LICENCE_ENV) == "1" and bfree_repo.is_dir():
            try:
                self.arms.append(BFreeArm(device, bfree_repo))
            except Exception as error:
                print(f"[verdict] B-Free yüklenemedi: {error}")
        elif bfree_repo.is_dir():
            print(f"[verdict] B-Free mevcut ama {LICENCE_ENV}=1 verilmedi "
                  "(lisans: yalnız araştırma/kâr amaçsız) — CF ile devam")
        self.primary = next(
            (a for a in self.arms if a.name == "bfree"), self.arms[0] if self.arms else None
        )
        self.load_seconds = time.perf_counter() - began

    @property
    def available(self) -> bool:
        return self.primary is not None

    def run(self, picture: Image.Image, byte_size: int) -> dict:
        width, height = picture.size
        bytes_per_pixel = byte_size / max(width * height, 1)
        scoring_input, was_capped = capped(picture)

        arms = {}
        for arm in self.arms:
            value = arm.score(scoring_input if arm.name == "bfree" else picture)
            arms[arm.name] = {
                "label": arm.label,
                "score": round(value, 4),
                "threshold": round(arm.threshold, 4),
                "band": decide(value, arm.threshold),
            }

        caveats = []
        if was_capped:
            caveats.append("megapiksel-siniri")      # E23b policy fired
        if bytes_per_pixel <= COMPRESSED_BPP:
            caveats.append("sikistirilmis-girdi")    # E23c degraded regime

        label, triggered = combine({n: a["band"] for n, a in arms.items()})
        return {
            "label": label,
            "triggered_by": triggered,
            "arms": arms,
            "caveats": caveats,
            "bytes_per_pixel": round(bytes_per_pixel, 3),
            "provenance": "eşikler: 12 kaynaklı kalibrasyon kütüphanesi, "
                          "worst-source + OR kuralı (E22/E23/E24/E26)",
        }
