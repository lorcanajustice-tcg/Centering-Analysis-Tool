"""Regression: the Gadget 12:147 white-paper pair shrunk to low resolution.

Before v0.4.0 a front below ~12 px/mm refused outright: the fine search
window is 2.5mm each side of the seed, and at 7 px/mm that is 18px - under
the 20px each side the brightness scanner needs, so every scan line came
back "the search ran off the side of the photo". The window now has a
pixel floor (edges.MIN_SCAN_HALF_WINDOW_PX).

Observed 2026-09-15 (front shift x / y, mm; full size is 33.4 px/mm):
    scale  px/mm   x                 y
    1.00   33.4   -0.103 +-0.065    +0.214 +-0.117
    0.30   10.0   -0.104 +-0.094    +0.181 +-0.135
    0.22    7.3   -0.132 +-0.114    +0.091 +-0.150
    0.15    5.0   -0.091 +-0.154    +0.023 +-0.182
All four edges "ok" at every scale; the back reads L/R 48-49, T/B 48-50
throughout, its uncertainty growing from 1.2 to 4.8 points as it should.
"""
from pathlib import Path

import pytest
from PIL import Image, ImageOps

from centering import analyze_back, analyze_borderless
from centering.games.lorcana import LORCANA

FIX = Path(__file__).resolve().parents[2] / "fixtures"
FRONT, BACK = FIX / "IMG_6397.HEIC", FIX / "IMG_6398.HEIC"
SCALE = 0.22

pytestmark = pytest.mark.skipif(
    not (FRONT.exists() and BACK.exists()),
    reason="Gadget white-paper fixture photos not present")


def _shrink(src, dst_dir):
    import centering.imgio  # noqa: F401  (registers the HEIC opener)
    im = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    im = im.resize((round(im.width * SCALE), round(im.height * SCALE)),
                   Image.LANCZOS)
    out = dst_dir / (src.stem + "_small.jpg")
    im.save(out, quality=92)
    return out


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    d = tmp_path_factory.mktemp("lowres")
    return _shrink(FRONT, d), _shrink(BACK, d)


def test_front_measures_at_7px_per_mm(small):
    r = analyze_borderless(small[0], "12:147", LORCANA, make_overlay=False)
    assert 6.5 < r.input.px_per_mm < 8.0
    assert all(e.status == "ok" for e in r.edge_fits)
    x, y = r.shift_mm["x"], r.shift_mm["y"]
    assert x.status == y.status == "measured"
    # full-size values -0.10 / +0.21, inside the low-res margins
    assert x.value == pytest.approx(-0.10, abs=max(0.1, x.uncertainty.total))
    assert y.value == pytest.approx(0.21, abs=max(0.15, y.uncertainty.total))
    assert x.uncertainty.total > 0.08     # and it knows it is coarser


def test_back_measures_at_7px_per_mm(small):
    b = analyze_back(small[1], LORCANA, make_overlay=False)
    assert b.ratio_lr.first_pct == pytest.approx(48.3, abs=2.0)
    assert b.ratio_tb.first_pct == pytest.approx(48.2, abs=2.5)
    assert b.ratio_lr.uncertainty_pts.total > 2.0
