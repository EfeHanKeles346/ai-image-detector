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
from pixelproof.e32_r1b_candidate import ARTIFACT_SHA256 as R1B_ARTIFACT_SHA256
from pixelproof.e32_r1b_candidate import E32R1bCandidate
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
from pixelproof.project_model import (
    ProjectImageScore,
    ProjectTileModel,
    load_project_model,
)
from pixelproof.verdict import VerdictService


CNN_ROUTING_PX = 128
TILE_RELIABLE_PX = 700
UNCERTAINTY_BAND = 0.1
MAX_TILES = 256

STATS_VARIANTS = {"stats": "full"}
METHODS = {
    "project_model": "Proje modeli — E20 yerel-tile ResNet-18",
    "auto": "Otomatik — görsel boyutuna göre en güçlü yöntem",
    "cnn": "CNN — küçültülmüş görsel, sinir ağı",
    "stats": "İstatistik — 68 ölçüm, 9.9k GenImage ile eğitildi",
    "tiles": "Kare kare — en fazla 256 yerel kesit",
}

CORE_ARTIFACT_IDS = {
    "small-cnn-cifake",
    "resnet18-genimage",
    "feature-full",
    "feature-crop128",
}
CF_ARTIFACT_ID = "community-forensics-vit-s"
R1B_ENV = "PIXELPROOF_R1B"
PROJECT_LIMITATION = (
    "Araştırma modeli: E20 üç-seed değerlendirmesinde en kötü gerçek kaynakta "
    "%86,2 yanlış pozitif verdi; sonuç gerçeklik sertifikası değildir."
)
R1B_LIMITATION = (
    "Araştırma adayı: bağımsız IPN testinde en kötü cihaz yanlış pozitifi %40,0; "
    "owner-gallery yanlış pozitifi %68,57. Ana kararı değiştirmez."
)


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

    def __init__(self, artifacts_dir: Path | None = None, profile: str | None = None) -> None:
        self.device = select_device()
        self.artifacts = (artifacts_dir or Path(__file__).resolve().parents[2] / "artifacts").resolve()
        self.profile = profile or os.environ.get("PIXELPROOF_RUNTIME_PROFILE", "full")
        if self.profile not in {"full", "demo", "project"}:
            raise ValueError("PIXELPROOF_RUNTIME_PROFILE must be 'full', 'demo' or 'project'")
        self.cnns: dict[str, tuple[Any, Any]] = {}
        self.features: dict[str, Any] = {}
        self.project_model: ProjectTileModel | None = None
        self.verdict: VerdictService | None = None
        self.r1b: E32R1bCandidate | None = None
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
                return self.available
            self.load_attempted = True

            try:
                self.project_model = load_project_model(self.artifacts.parent, self.device)
            except Exception as error:
                self.load_errors["project_model"] = f"{type(error).__name__}: {error}"

            if self.profile == "project":
                return self.available

            if self.profile == "full":
                try:
                    core_report = verify_registry(self.artifacts.parent, groups={"core"})
                except Exception as error:
                    self.load_errors["artifact_manifest"] = f"{type(error).__name__}: {error}"
                    core_report = {"ok": False, "checked": [], "issues": []}
                checked_core = set(core_report["checked"])
                if not core_report["ok"]:
                    if core_report["issues"]:
                        self.load_errors["artifact_manifest"] = "; ".join(core_report["issues"])
                elif checked_core != CORE_ARTIFACT_IDS:
                    missing = sorted(CORE_ARTIFACT_IDS - checked_core)
                    self.load_errors["artifact_manifest"] = (
                        f"core manifest entries missing: {', '.join(missing)}"
                    )

                if "artifact_manifest" not in self.load_errors:
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
                if cf_report["ok"] and CF_ARTIFACT_ID in cf_report["checked"]:
                    self.verdict = VerdictService(self.device, self.artifacts.parent)
                else:
                    issues = cf_report["issues"] or [f"manifest entry missing: {CF_ARTIFACT_ID}"]
                    self.load_errors["cf_vit_manifest"] = "; ".join(issues)
            except Exception as error:
                self.load_errors["verdict"] = f"{type(error).__name__}: {error}"

            if os.environ.get(R1B_ENV) == "1":
                try:
                    shared = next(
                        (arm for arm in (self.verdict.arms if self.verdict else []) if arm.name == "cf_vit"),
                        None,
                    )
                    if shared is None:
                        raise RuntimeError("shared CF-ViT arm unavailable")
                    self.r1b = E32R1bCandidate(
                        device=self.device,
                        model=shared.model,
                        processor=shared.processor,
                    )
                except Exception as error:
                    self.load_errors["r1b_research"] = f"{type(error).__name__}: {error}"
            return self.available

    @property
    def project_model_ready(self) -> bool:
        return self.project_model is not None

    @property
    def core_ready(self) -> bool:
        return (
            {"small_cnn_cifake", "resnet18_genimage"}.issubset(self.cnns)
            and {"full", "crop128"}.issubset(self.features)
        )

    @property
    def decision_ready(self) -> bool:
        return self.verdict is not None and self.verdict.available

    @property
    def available(self) -> bool:
        return self.project_model_ready or self.core_ready or self.decision_ready

    def health(self) -> dict[str, Any]:
        if not self.load_attempted:
            status = "starting"
        elif self.project_model_ready:
            status = "ready"
        elif self.core_ready or self.decision_ready:
            status = "degraded"
        else:
            status = "unavailable"

        arms: list[str] = []
        if self.verdict is not None:
            arms.extend(arm.name for arm in self.verdict.arms)
        return {
            "status": status,
            "device": str(self.device),
            "runtime_profile": self.profile,
            "project_model_ready": self.project_model_ready,
            "project_model": (
                self.project_model.metadata.to_dict() if self.project_model is not None else None
            ),
            "core_ready": self.core_ready,
            "decision_ready": self.decision_ready,
            "r1b_research_ready": self.r1b is not None,
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

    def _run_project_model(
        self,
        picture: Image.Image,
    ) -> tuple[float, dict[str, Any], dict[str, Any]]:
        if self.project_model is None:
            detail = self.load_errors.get("project_model", "canonical E20 artifact unavailable")
            raise RuntimeUnavailable(f"Proje modeli kullanılamıyor: {detail}")
        measured: ProjectImageScore = self.project_model.score_image(
            picture,
            max_tiles=MAX_TILES,
        )
        metadata = self.project_model.metadata
        project_payload = {
            "score": round(measured.score, 4),
            "threshold": round(metadata.threshold, 4),
            "triggered": measured.triggered,
            "research_only": True,
            "limitation": PROJECT_LIMITATION,
            "artifact_id": metadata.artifact_id,
            "artifact_sha256": metadata.sha256,
            "revision": metadata.revision,
            "seed": metadata.seed,
            "aggregation": metadata.aggregation,
            "tile_px": metadata.tile_px,
            "tile_count": measured.tile_count,
        }
        tile_map = {
            "p_ai": round(measured.score, 4),
            "tiles": [
                {
                    "x": x,
                    "y": y,
                    "p_ai": round(score, 4),
                    "texture": round(texture, 4),
                }
                for (x, y), score, texture in zip(
                    measured.positions,
                    measured.tile_scores,
                    measured.textures,
                )
            ],
            "tile_px": metadata.tile_px,
            "image_w": measured.width,
            "image_h": measured.height,
        }
        return measured.score, project_payload, tile_map

    def _run_r1b(self, picture: Image.Image) -> dict[str, Any] | None:
        if self.r1b is None:
            return None
        score = self.r1b.score_image(picture)
        triggered = score >= self.r1b.threshold
        return {
            "id": "e32_r1b_cfvit_iphone_correction",
            "label": "E32 R1b · CF-ViT",
            "score": round(score, 4),
            "threshold": round(self.r1b.threshold, 4),
            "triggered": triggered,
            "band": "ai_signal" if triggered else "insufficient_evidence",
            "research_only": True,
            "affects_decision": False,
            "artifact_sha256": R1B_ARTIFACT_SHA256,
            "limitation": R1B_LIMITATION,
            "evaluation": {
                "ipn_worst_device_fp": 0.4,
                "owner_gallery_fp": 144 / 210,
            },
        }

    def predict(self, picture: Image.Image, byte_size: int, method: str) -> dict[str, Any]:
        self.ensure_loaded()

        with self._inference_slot:
            chosen = method
            if method == "auto":
                chosen = "tiles" if max(picture.size) >= TILE_RELIABLE_PX else "cnn"

            tile_map = None
            project_payload = None
            if chosen == "project_model":
                score, project_payload, tile_map = self._run_project_model(picture)
                engine = self.project_model.metadata.artifact_id
            elif not self.core_ready:
                detail = self.load_errors.get(
                    "artifact_manifest",
                    "legacy research artifacts unavailable",
                )
                raise RuntimeUnavailable(f"Eski araştırma yöntemi kullanılamıyor: {detail}")
            elif chosen == "cnn":
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
            try:
                r1b_research = self._run_r1b(picture)
            except Exception as error:
                self.load_errors["r1b_inference"] = f"{type(error).__name__}: {error}"
                r1b_research = None
            width, height = picture.size
            return {
                "p_ai": round(score, 4),
                "verdict": (
                    "ai" if project_payload and project_payload["triggered"] else
                    "uncertain" if project_payload else verdict_for(score)
                ),
                "method": chosen,
                "method_label": METHODS[chosen],
                "auto_selected": method == "auto",
                "engine": engine,
                "resolution": f"{width}x{height}",
                "enough_evidence": True,
                "tile_map": tile_map,
                "project_model": project_payload,
                "decision": decision,
                "r1b_research": r1b_research,
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
        method: str = Form("project_model"),
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
