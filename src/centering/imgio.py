"""Image loading: EXIF orientation, HEIC support, hashing."""
from __future__ import annotations

import hashlib
from pathlib import Path

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
