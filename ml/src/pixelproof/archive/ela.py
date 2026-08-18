"""Phase 7a: Error Level Analysis (ELA) — the classical baseline of Module 2.

Re-saving an image as JPEG degrades every region a little. Regions that were
pasted in or regenerated later carry a different compression history, so they
change MORE on re-save than the rest of the picture. The difference map makes
those regions glow.

Usage: PYTHONPATH=src .venv/bin/python -m pixelproof.ela photo.jpg
Output: <photo>_ela.jpg — original on the left, ELA map on the right.
"""

import sys
import tempfile
from pathlib import Path

from PIL import Image, ImageChops, ImageEnhance

JPEG_QUALITY = 90


def ela_map(image: Image.Image, quality: int = JPEG_QUALITY) -> Image.Image:
    with tempfile.NamedTemporaryFile(suffix=".jpg") as temp:
        image.save(temp.name, "JPEG", quality=quality)
        resaved = Image.open(temp.name).convert("RGB")
        difference = ImageChops.difference(image, resaved)
    # Raw differences are tiny (single-digit pixel values); stretch the
    # brightest one up to 255 so the map is visible to the eye.
    peak = max(channel_max for _, channel_max in difference.getextrema()) or 1
    return ImageEnhance.Brightness(difference).enhance(255.0 / peak)


def main() -> None:
    path = Path(sys.argv[1])
    original = Image.open(path).convert("RGB")
    heatmap = ela_map(original)

    side_by_side = Image.new("RGB", (original.width * 2, original.height))
    side_by_side.paste(original, (0, 0))
    side_by_side.paste(heatmap, (original.width, 0))
    output = path.with_name(f"{path.stem}_ela.jpg")
    side_by_side.save(output, quality=95)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
