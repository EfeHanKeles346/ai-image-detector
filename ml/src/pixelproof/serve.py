"""Bounded FastAPI inference service for the PixelProof research demo.

Run locally:
    PYTHONPATH=src .venv/bin/uvicorn pixelproof.serve:app --port 8799

The four research methods are never blended. The E22-E27 decision layer remains
asymmetric (``ai`` or ``insufficient``) and may be unavailable when its external
artifacts are absent; health reports that state instead of failing at import time.
"""

from __future__ import annotations

import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Protocol

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from starlette.concurrency import run_in_threadpool

from pixelproof.artifact_registry import verify_registry
from pixelproof.evaluate import eval_transform
from pixelproof.feature_model import load as load_feature_models
from pixelproof.feature_model import score_image, score_tiles
from pixelproof.image_input import (
    DEFAULT_LIMITS,
    ImageLimits,
    ImagePolicyError,
    decode_image,
    enough_evidence,
)
from pixelproof.models import create_model
from pixelproof.verdict import VerdictService


CNN_ROUTING_PX = 128
TILE_RELIABLE_PX = 700
UNCERTAINTY_BAND = 0.1
MAX_TILES = 256

STATS_VARIANTS = {"stats": "full"}
METHODS = {
    "auto": "Otomatik — görsel boyutuna göre en güçlü yöntem",
    "cnn": "CNN — küçültülmüş görsel, sinir ağı",
    "stats": "İstatistik — 68 ölçüm, 9.9k GenImage ile eğitildi",
    "tiles": "Kare kare — en fazla 256 yerel kesit",
}


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def verdict_for(score: float) -> str:
    if abs(score - 0.5) < UNCERTAINTY_BAND:
        return "uncertain"
    return "ai" if score >= 0.5 else "real"


class RuntimeUnavailable(RuntimeError):
    pass


class RuntimeContract(Protocol):
    def ensure_loaded(self) -> bool: ...
    def health(self) -> dict[str, Any]: ...
    def predict(self, picture: Image.Image, byte_size: int, method: str) -> dict[str, Any]: ...


class ModelRuntime:
    """Owns model state and serializes expensive inference on one worker."""

    def __init__(self, artifacts_dir: Path | None = None) -> None:
        self.device = select_device()
        self.artifacts = (artifacts_dir or Path(__file__).resolve().parents[2] / "artifacts").resolve()
        self.cnns: dict[str, tuple[Any, Any]] = {}
        self.features: dict[str, Any] = {}
        self.verdict: VerdictService | None = None
        self.load_errors: dict[str, str] = {}
        self.load_attempted = False
        self._load_lock = threading.Lock()
        self._inference_slot = threading.BoundedSemaphore(value=1)

    def _load_checkpoint(self, checkpoint_name: str) -> tuple[Any, Any]:
        checkpoint = torch.load(
            self.artifacts / checkpoint_name,
            map_location=self.device,
            weights_only=False,
        )
        config = checkpoint["config"]
        model = create_model(
            config["model"]["name"],
            dropout=config["model"]["dropout"],
        ).to(self.device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        transform = eval_transform(
            config["data"]["image_size"],
            config["data"].get("normalization", "default"),
        )
        return model, transform

    def ensure_loaded(self) -> bool:
        with self._load_lock:
            if self.load_attempted:
                return self.core_ready
            self.load_attempted = True

            try:
                core_report = verify_registry(self.artifacts.parent, groups={"core"})
            except Exception as error:
                self.load_errors["artifact_manifest"] = f"{type(error).__name__}: {error}"
                return False
            if not core_report["ok"]:
                self.load_errors["artifact_manifest"] = "; ".join(core_report["issues"])
                return False

            for name, filename in {
                "small_cnn_cifake": "best.pt",
                "resnet18_genimage": "best_genimage.pt",
            }.items():
                try:
                    self.cnns[name] = self._load_checkpoint(filename)
                except Exception as error:
                    self.load_errors[name] = f"{type(error).__name__}: {error}"

            try:
                self.features = load_feature_models(self.artifacts)
            except Exception as error:
                self.load_errors["features"] = f"{type(error).__name__}: {error}"

            try:
                cf_report = verify_registry(self.artifacts.parent, groups={"cf_vit"})
                if cf_report["ok"]:
                    self.verdict = VerdictService(self.device, self.artifacts.parent)
                else:
                    self.load_errors["cf_vit_manifest"] = "; ".join(cf_report["issues"])
            except Exception as error:
                self.load_errors["verdict"] = f"{type(error).__name__}: {error}"
            return self.core_ready

    @property
    def core_ready(self) -> bool:
        return (
            {"small_cnn_cifake", "resnet18_genimage"}.issubset(self.cnns)
            and {"full", "crop128"}.issubset(self.features)
        )

    @property
    def decision_ready(self) -> bool:
        return self.verdict is not None and self.verdict.available

    def health(self) -> dict[str, Any]:
        if not self.load_attempted:
            status = "starting"
        elif not self.core_ready:
            status = "unavailable"
        elif not self.decision_ready:
            status = "degraded"
        else:
            status = "ready"

        arms: list[str] = []
        if self.verdict is not None:
            arms.extend(arm.name for arm in self.verdict.arms)
        return {
            "status": status,
            "device": str(self.device),
            "core_ready": self.core_ready,
            "decision_ready": self.decision_ready,
            "cnns": sorted(self.cnns),
            "features": sorted(self.features),
            "verdict_arms": arms,
            "verdict_rule": "OR (E26)",
            "load_errors": self.load_errors,
        }

    def _run_cnn(self, picture: Image.Image) -> tuple[float, str]:
        name = "small_cnn_cifake" if max(picture.size) < CNN_ROUTING_PX else "resnet18_genimage"
        model, transform = self.cnns[name]
        with torch.inference_mode():
            score = torch.sigmoid(
                model(transform(picture).unsqueeze(0).to(self.device))
            ).item()
        return score, name

    def predict(self, picture: Image.Image, byte_size: int, method: str) -> dict[str, Any]:
        if not self.ensure_loaded():
            raise RuntimeUnavailable("Temel model artifact'leri yüklenemedi; /health ayrıntılarını kontrol edin.")

        with self._inference_slot:
            chosen = method
            if method == "auto":
                chosen = "tiles" if max(picture.size) >= TILE_RELIABLE_PX else "cnn"

            tile_map = None
            if chosen == "cnn":
                score, engine = self._run_cnn(picture)
            elif chosen in STATS_VARIANTS:
                variant = STATS_VARIANTS[chosen]
                score = score_image({variant: self.features[variant]}, picture)[variant]
                engine = f"feature_{variant}"
            else:
                tile_map = score_tiles(self.features, picture, grid=MAX_TILES)
                if tile_map is None:
                    raise RuntimeUnavailable("Kare modeli yüklenemedi.")
                score = tile_map["p_ai"]
                engine = "feature_tiles"

            decision = self.verdict.run(picture, byte_size) if self.decision_ready else None
            width, height = picture.size
            return {
                "p_ai": round(score, 4),
                "verdict": verdict_for(score),
                "method": chosen,
                "method_label": METHODS[chosen],
                "auto_selected": method == "auto",
                "engine": engine,
                "resolution": f"{width}x{height}",
                "enough_evidence": True,
                "tile_map": tile_map,
                "decision": decision,
            }


def configured_origins(value: str | None = None) -> list[str]:
    raw = value if value is not None else os.environ.get(
        "PIXELPROOF_CORS_ORIGINS",
        "http://localhost:3000,http://127.0.0.1:3000",
    )
    origins = [item.strip().rstrip("/") for item in raw.split(",") if item.strip()]
    if "*" in origins:
        raise ValueError("PIXELPROOF_CORS_ORIGINS wildcard kullanamaz.")
    return origins


def create_app(
    runtime: RuntimeContract | None = None,
    limits: ImageLimits = DEFAULT_LIMITS,
    allowed_origins: list[str] | None = None,
) -> FastAPI:
    active_runtime = runtime or ModelRuntime()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        await run_in_threadpool(active_runtime.ensure_loaded)
        yield

    application = FastAPI(title="PixelProof inference", lifespan=lifespan)
    origins = configured_origins() if allowed_origins is None else allowed_origins
    if origins:
        application.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Content-Type"],
        )

    @application.post("/predict")
    async def predict_endpoint(
        image: UploadFile = File(...),
        method: str = Form("auto"),
    ):
        if method not in METHODS:
            raise HTTPException(422, f"Bilinmeyen yöntem: {method}")

        raw = await image.read(limits.max_upload_bytes + 1)
        if len(raw) > limits.max_upload_bytes:
            raise HTTPException(
                413,
                f"Dosya {limits.max_upload_bytes // (1024 * 1024)} MB sınırını aşıyor.",
            )
        try:
            picture = await run_in_threadpool(decode_image, raw, limits)
        except ImagePolicyError as error:
            raise HTTPException(error.status_code, error.detail) from error

        sufficient = enough_evidence(picture, limits)
        try:
            result = await run_in_threadpool(
                active_runtime.predict,
                picture,
                len(raw),
                method,
            )
        except RuntimeUnavailable as error:
            raise HTTPException(503, str(error)) from error
        except Exception as error:
            raise HTTPException(500, "Model çıkarımı başarısız oldu.") from error

        width, height = picture.size
        result["resolution"] = f"{width}x{height}"
        result["enough_evidence"] = sufficient
        if not sufficient:
            result["decision"] = None
        return result

    @application.get("/methods")
    def methods_endpoint():
        return {
            "methods": [{"id": key, "label": value} for key, value in METHODS.items()],
            "tile_reliable_px": TILE_RELIABLE_PX,
            "max_tiles": MAX_TILES,
        }

    @application.get("/health")
    def health_endpoint():
        return active_runtime.health()

    return application


app = create_app()
