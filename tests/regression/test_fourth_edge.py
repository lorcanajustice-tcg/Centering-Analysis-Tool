"""Regression: a hand-drawn box that misses one edge (fourth-edge path).

Photo: Belle & Beast - Captain of the Sun 245/207 EN (set 13, card id
"13-245"), a 540x960 third-party phone photo of the card in a one-touch
case on a light background, only 6.9-7.5 px/mm. Lives in the failure
corpus, NOT in fixtures/: `Training Data/real failures/IMG_2033.jpg`
beside the repository (git-ignored card imagery; the test skips without
it).

Why it matters: automatic localization grabs the case, so the photo needs
the manual drag-box. The box the user drew runs ~7mm past the card's
bottom edge, so the first bottom scan searched the case and found 6 usable
lines of 50 - and one missing edge used to refuse BOTH axes.

What the fourth-edge path does with it (2026-09-15, observed):
- builds the bottom from left/right/top and the card's 87.9mm height,
  6.7mm from where the box put it;
- looks again THERE and measures it for real (step, 41-44 lines, rms
  ~0.3px, within 0.7mm of the prediction) -> EDGE_RELOCATED;
- top-to-bottom is then MEASURED: shift y ~-0.09mm, T/B ~48/52;
- left-to-right stays REFUSED by the render-span gate, correctly: the
  right edge fit leans (slope 0.075 where the card's right edge leans
  ~0.045) and was never trustworthy. The fourth-edge path must not
  paper over that.
Bounds are loose on purpose (low resolution, weak render match: ~144
inliers).
"""
from pathlib import Path

import pytest

from centering import analyze_borderless
from centering.games.lorcana import LORCANA

PHOTO = (Path(__file__).resolve().parents[3] / "Training Data"
         / "real failures" / "IMG_2033.jpg")
BOX = (52, 168, 492, 884)

pytestmark = pytest.mark.skipif(not PHOTO.exists(),
                                reason="failure-corpus photo not present")


@pytest.fixture(scope="module")
def res():
    return analyze_borderless(PHOTO, "13-245", LORCANA, make_overlay=False,
                              manual_bbox=BOX)


def _edge(res, side):
    return next(e for e in res.edge_fits if e.edge == side)


def test_bottom_relocated_and_measured(res):
    b = _edge(res, "bottom")
    assert b.method in ("step", "texture", "colour")
    assert b.status in ("ok", "flagged")
    assert b.n_points >= 25
    assert b.rms_residual_px <= 1.0
    assert "second look" in " ".join(b.notes)
    codes = [q.code for q in res.qa]
    assert "EDGE_RELOCATED" in codes
    assert "EDGE_INFERRED" not in codes


def test_top_to_bottom_is_measured(res):
    y = res.shift_mm["y"]
    assert y.status == "measured"
    assert abs(y.value) < 0.35
    assert y.uncertainty.total < 0.3
    assert 40 < res.equivalent_ratio_tb.first_pct < 56


def test_left_to_right_still_refused_by_span_gate(res):
    assert res.shift_mm["x"].status == "refused"
    assert any(q.code == "RENDER_SPAN_MISMATCH" for q in res.qa)
