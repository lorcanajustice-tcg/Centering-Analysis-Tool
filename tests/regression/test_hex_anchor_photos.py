"""Regression: the ink-cost hexagon check on real phone photos.

Fixtures (git-ignored, see fixtures/README.md): IMG_6341 (Elsa 6/C2,
uninkable, dark mat), IMG_6397 / IMG_6401 (Gadget 12:147, inkable, white
paper / kraft), IMG_6330 (Simba 8/C2, uninkable, dark mat).

Known-good values were set on 2026-09-16 when the check was added. Across
the eight clean fronts of the 2026-07-06 reshoot and the fixtures, the
hexagon and the official-picture match agreed to 0.034mm rms left-right
and 0.088mm top-bottom; corner close-ups cut from the same photos to
0.041 / 0.102mm.
"""
from pathlib import Path

import cv2
import pytest

from centering.borderless import analyze_borderless
from centering.games.lorcana import LORCANA
from centering.hexanchor import analyze_corner
from centering.imgio import load_photo

FIX = Path(__file__).resolve().parents[2] / "fixtures"

# photo, card, official-picture shift (x, y) mm
CASES = [
    ("IMG_6341.HEIC", "6/C2", "uninkable", (-0.244, 0.289)),
    ("IMG_6397.HEIC", "12:147", "inkable", (-0.122, 0.142)),
    ("IMG_6401.HEIC", "12:147", "inkable", (-0.122, -0.001)),
]


def _need(name):
    p = FIX / name
    if not p.exists():
        pytest.skip(f"fixture {name} not present")
    return p


@pytest.mark.parametrize("photo,card,layout,ref", CASES)
def test_cross_check_agrees_with_the_official_picture(photo, card, layout,
                                                      ref, tmp_path):
    r = analyze_borderless(_need(photo), card, LORCANA, out_dir=str(tmp_path),
                           make_overlay=False)
    hx = r.hex_check
    assert hx.status == "measured"
    assert hx.layout == layout
    assert "own entry in the survey" in hx.layout_source
    assert abs(hx.agreement_mm["x"]) < 0.08
    assert abs(hx.agreement_mm["y"]) < 0.15
    assert not any(q.code == "HEX_ANCHOR_DISAGREES" for q in r.qa)
    for ax in "xy":
        assert abs(hx.size_vs_card_pct[ax]) < 1.0


@pytest.mark.parametrize("photo,card,layout,ref", CASES)
def test_front_without_card_id(photo, card, layout, ref, tmp_path):
    r = analyze_borderless(_need(photo), None, LORCANA, out_dir=str(tmp_path),
                           make_overlay=False)
    assert r.method == "ink_hexagon"
    assert r.hex_check.layout == layout
    assert r.shift_mm["x"].value == pytest.approx(ref[0], abs=0.08)
    assert r.shift_mm["y"].value == pytest.approx(ref[1], abs=0.15)
    # an unknown uninkable card carries the sets 1-3 note
    notes = [q.code for q in r.qa]
    assert ("HEX_LAYOUT_ASSUMED" in notes) == (layout == "uninkable")


# corner crops, as (left, top, right, bottom) px of the full photo
CROPS = {
    "IMG_6397.HEIC": (250, 335, 1418, 1503),
    "IMG_6341.HEIC": (55, 31, 1444, 1420),
}


@pytest.mark.parametrize("photo,card,layout,ref",
                         [c for c in CASES if c[0] in CROPS])
def test_corner_close_up(photo, card, layout, ref, tmp_path):
    rgb, _, _ = load_photo(_need(photo))
    x0, y0, x1, y1 = CROPS[photo]
    p = tmp_path / (Path(photo).stem + "_corner.png")
    cv2.imwrite(str(p), cv2.cvtColor(rgb[y0:y1, x0:x1], cv2.COLOR_RGB2BGR))
    r = analyze_corner(p, LORCANA, make_overlay=False)
    assert r.hex_check.layout == layout
    for ax, v in zip("xy", ref):
        m = r.shift_mm[ax]
        assert m.status == "measured"
        # within the quoted margin of the official-picture value
        assert abs(m.value - v) < m.uncertainty.total
