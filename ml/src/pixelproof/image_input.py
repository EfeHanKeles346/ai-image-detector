"""Bounded, deterministic image decoding shared by every served detector arm."""

from __future__ import annotations

import io
import warnings
from dataclasses import dataclass

from PIL import Image, ImageOps, UnidentifiedImageError


SUPPORTED_FORMATS = frozenset({"JPEG", "PNG", "WEBP"})
TRANSPARENCY_BACKGROUND = (255, 255, 255, 255)


@dataclass(frozen=True)
class ImageLimits:
    max_upload_bytes: int = 12 * 1024 * 1024
    max_pixels: int = 16_000_000
    max_dimension: int = 16_384
    max_aspect_ratio: float = 20.0
    evidence_floor_px: int = 48


DEFAULT_LIMITS = ImageLimits()


class ImagePolicyError(ValueError):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail


def enough_evidence(picture: Image.Image, limits: ImageLimits = DEFAULT_LIMITS) -> bool:
    """Both axes need measured support; a 4x4000 strip is not a 4000px image."""
    width, height = picture.size
    return min(width, height) >= limits.evidence_floor_px


def _validate_geometry(width: int, height: int, limits: ImageLimits) -> None:
    if width <= 0 or height <= 0:
        raise ImagePolicyError(422, "Görsel boyutları geçersiz.")
    if width > limits.max_dimension or height > limits.max_dimension:
        raise ImagePolicyError(
            413,
            f"Görselin bir kenarı {limits.max_dimension} piksel sınırını aşıyor.",
        )
    if width * height > limits.max_pixels:
        raise ImagePolicyError(
            413,
            f"Görsel {limits.max_pixels:,} piksel sınırını aşıyor.",
        )
    if max(width, height) / min(width, height) > limits.max_aspect_ratio:
        raise ImagePolicyError(
            422,
            f"En-boy oranı {limits.max_aspect_ratio:g}:1 sınırını aşıyor.",
        )


def decode_image(raw: bytes, limits: ImageLimits = DEFAULT_LIMITS) -> Image.Image:
    """Validate headers, fully decode, apply EXIF orientation, then flatten alpha on white."""
    if not raw:
        raise ImagePolicyError(422, "Yüklenen dosya boş.")
    if len(raw) > limits.max_upload_bytes:
        raise ImagePolicyError(
            413,
            f"Dosya {limits.max_upload_bytes // (1024 * 1024)} MB sınırını aşıyor.",
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(raw)) as source:
                if source.format not in SUPPORTED_FORMATS:
                    raise ImagePolicyError(
                        415,
                        "Yalnız JPG, PNG ve WEBP biçimleri destekleniyor.",
                    )
                _validate_geometry(*source.size, limits)
                source.load()
                oriented = ImageOps.exif_transpose(source)
                _validate_geometry(*oriented.size, limits)

                if oriented.mode in {"RGBA", "LA"} or "transparency" in oriented.info:
                    foreground = oriented.convert("RGBA")
                    background = Image.new("RGBA", foreground.size, TRANSPARENCY_BACKGROUND)
                    background.alpha_composite(foreground)
                    return background.convert("RGB")
                return oriented.convert("RGB")
    except ImagePolicyError:
        raise
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ImagePolicyError(413, "Görsel güvenli piksel sınırını aşıyor.") from None
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        raise ImagePolicyError(415, "Dosya geçerli bir JPG, PNG veya WEBP görseli değil.") from None
