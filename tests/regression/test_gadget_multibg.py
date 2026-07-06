"""Regression: Gadget Hackwrench - Finder of Lost Parts 147/204 EN (set 12,
card id "12:147"), front+back photographed on THREE backgrounds
(IMG_6397-6402, shot 2026-07-06): white paper, dark mat, kraft cardboard.

Validates the polarity-agnostic pipeline (2026-07-06): white and kraft
backgrounds must yield full measurements via the polarity-agnostic
brightness-step scanners, and the same physical card must measure
consistently across backgrounds.

Capture conditions / known artifacts (kept deliberately as honest-refusal
cases - do NOT "fix" these bounds by weakening refusals):
- dark-mat back (IMG_6400): shadow along the bottom edge -> bottom border
  and T/B refused (documented dark-mat artifact);
- dark-mat front (IMG_6399): directional light (BACKGROUND_NONUNIFORM);
  right border has neither step nor texture signal, top edge partially
  glare-banded -> gross aspect deviation -> shifts REFUSED with an
  ASPECT_DEVIATION QA flag rather than reported biased.

Known-good values at introduction (LOOSE bounds per test conventions):
back  white L 2.31 R 2.46 T 2.12 B 2.28  (L/R 48.3, T/B 48.2)
back  kraft L 2.26 R 2.44 T 2.13 B 2.22  (L/R 48.1, T/B 49.0)
back  dark  L/R 47.2, bottom refused
front white x -0.12mm  y -0.14mm; front kraft x -0.12mm y -0.28mm
front-vs-back x registration agrees within 0.05mm on white.
"""
from pathlib import Path

import pytest

from centering import analyze_back, analyze_borderless
from centering.games.lorcana import LORCANA

FIX = Path(__file__).resolve().parents[2] / "fixtures"
PHOTOS = {
    "front_white": FIX / "IMG_6397.HEIC",
    "back_white": FIX / "IMG_6398.HEIC",
    "front_dark": FIX / "IMG_6399.HEIC",
    "back_dark": FIX / "IMG_6400.HEIC",
    "front_kraft": FIX / "IMG_6401.HEIC",
    "back_kraft": FIX / "IMG_6402.HEIC",
}
CARD = "12:147"

pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in PHOTOS.values()),
    reason="Gadget multi-background fixture photos not present")


@pytest.fixture(scope="module")
def backs(tmp_path_factory):
    out = tmp_path_factory.mktemp("overlays")
    return {bg: analyze_back(PHOTOS[f"back_{bg}"], LORCANA, out_dir=out)
            for bg in ("white", "dark", "kraft")}


@pytest.fixture(scope="module")
def fronts(tmp_path_factory):
    out = tmp_path_factory.mktemp("overlays")
    return {bg: analyze_borderless(PHOTOS[f"front_{bg}"], CARD, LORCANA,
                                   out_dir=out)
            for bg in ("white", "dark", "kraft")}


# ---------------- backs ----------------

def test_back_white_fully_measured_via_step(backs):
    r = backs["white"]
    for s, v in (("left", 2.31), ("right", 2.46), ("top", 2.12),
                 ("bottom", 2.28)):
        m = r.borders_mm[s]
        assert m.status == "measured"
        assert m.value == pytest.approx(v, abs=0.12)
    fits = {e.edge: e for e in r.edge_fits if not e.edge.startswith("frame")}
    assert all(fits[s].method == "step" for s in fits)
    assert r.ratio_lr.first_pct == pytest.approx(48.3, abs=1.5)
    assert r.ratio_tb.first_pct == pytest.approx(48.2, abs=1.5)


def test_back_kraft_fully_measured(backs):
    r = backs["kraft"]
    assert all(r.borders_mm[s].status == "measured"
               for s in ("left", "right", "top", "bottom"))
    assert r.ratio_lr.first_pct == pytest.approx(48.1, abs=1.5)
    assert r.ratio_tb.first_pct == pytest.approx(49.0, abs=1.5)


def test_back_dark_lr_measured_bottom_refused(backs):
    r = backs["dark"]
    assert r.ratio_lr.first_pct == pytest.approx(47.2, abs=1.5)
    assert r.borders_mm["bottom"].status == "refused"
    assert r.ratio_tb.status == "refused"


def test_back_cross_background_consistency(backs):
    """Same physical card: L/R must agree across all three backgrounds."""
    pcts = [backs[bg].ratio_lr.first_pct for bg in ("white", "dark", "kraft")]
    assert max(pcts) - min(pcts) <= 1.5
    lefts = [backs[bg].borders_mm["left"].value
             for bg in ("white", "dark", "kraft")]
    assert max(lefts) - min(lefts) <= 0.12


# ---------------- fronts ----------------

def test_front_white_shift(fronts):
    r = fronts["white"]
    assert r.shift_mm["x"].status == "measured"
    assert r.shift_mm["x"].value == pytest.approx(-0.12, abs=0.10)
    assert r.shift_mm["y"].status == "measured"
    assert r.shift_mm["y"].value == pytest.approx(-0.14, abs=0.25)
    assert r.equivalent_ratio_lr.display
    assert r.render.n_inliers >= 200


def test_front_kraft_shift(fronts):
    r = fronts["kraft"]
    assert r.shift_mm["x"].value == pytest.approx(-0.12, abs=0.10)
    assert r.shift_mm["y"].value == pytest.approx(-0.28, abs=0.25)


def test_front_cross_background_consistency(fronts):
    dx = abs(fronts["white"].shift_mm["x"].value
             - fronts["kraft"].shift_mm["x"].value)
    assert dx <= 0.08


def test_front_dark_refuses_with_aspect_flag(fronts):
    """Directional-light dark-mat capture: gross aspect deviation must lead
    to an honest refusal, never a silently biased shift."""
    r = fronts["dark"]
    assert r.shift_mm["x"].status == "refused"
    assert r.shift_mm["y"].status == "refused"
    assert "ASPECT_DEVIATION" in [f.code for f in r.qa]
    assert r.equivalent_ratio_lr.status == "refused"


def test_front_back_x_registration_white(backs, fronts):
    """Front print shift and back border asymmetry describe the same die
    cut: implied x offsets must agree (target <=0.08mm registration)."""
    b = backs["white"].borders_mm
    back_implied_x = (b["left"].value - b["right"].value) / 2.0
    front_x = fronts["white"].shift_mm["x"].value
    assert abs(front_x - back_implied_x) <= 0.10
