"""Image loading: EXIF orientation, HEIC support, hashing."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageOps

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:  # pragma: no cover
    pass

from .types import InputReport


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_photo(path: str | Path) -> tuple[np.ndarray, np.ndarray, InputReport]:
    """Load at full resolution with EXIF orientation applied.

    Returns (rgb uint8 HxWx3, gray float32 HxW, InputReport).
    """
    path = Path(path)
    im = Image.open(path)
    orient = im.getexif().get(274, 1)
    im = ImageOps.exif_transpose(im)
    rgb = np.asarray(im.convert("RGB"))
    gray = np.asarray(im.convert("L"), dtype=np.float32)
    rep = InputReport(photo=path.name, sha256=sha256_file(path),
                      width=rgb.shape[1], height=rgb.shape[0],
                      exif_transposed=(orient != 1))
    return rgb, gray, rep


# A camera or scanner never produces a background of one exact colour:
# sensor noise and compression spread it over hundreds of values (real
# fixtures: the commonest single colour covers at most 4% of the photo's
# outer strip). A digital card picture padded onto a plain background
# covers 100%. Pure white is left out - an over-exposed sheet of paper
# really can clip to it.
DIGITAL_BORDER_FRAC = 0.6


def digital_background(rgb: np.ndarray) -> Optional[tuple]:
    """(colour, fraction) when the photo's outer strip is one exact colour
    - the mark of a digital image rather than a photo - else None."""
    if rgb is None or rgb.ndim != 3:
        return None
    h, w = rgb.shape[:2]
    b = max(2, int(0.01 * min(h, w)))
    strip = np.concatenate([rgb[:b].reshape(-1, 3), rgb[-b:].reshape(-1, 3),
                            rgb[:, :b].reshape(-1, 3),
                            rgb[:, -b:].reshape(-1, 3)])
    packed = (strip[:, 0].astype(np.int32) << 16) | \
        (strip[:, 1].astype(np.int32) << 8) | strip[:, 2].astype(np.int32)
    vals, cnt = np.unique(packed, return_counts=True)
    i = int(cnt.argmax())
    frac = float(cnt[i]) / len(packed)
    colour = ((int(vals[i]) >> 16) & 255, (int(vals[i]) >> 8) & 255,
              int(vals[i]) & 255)
    if frac < DIGITAL_BORDER_FRAC or min(colour) >= 250:
        return None
    return colour, frac


def digital_image_reason(found) -> str:
    colour, frac = found
    shade = "black" if max(colour) <= 10 else f"colour {colour}"
    return (f"This looks like a digital picture of the card, not a photo of "
            f"a real one: {frac * 100:.0f}% of the picture's outer edge is "
            f"exactly the same {shade}, which a camera or scanner never "
            "produces. A digital picture has no real cut edges - where the "
            "card's border meets a matching background there is nothing to "
            "find - and an official card picture is cropped the same way "
            "every time, so it says nothing about how a printed card was "
            "cut.")
