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

What the missing-edge path does with it (observed, 2026-09-15, v0.4.0):
- the first right-edge fit (13 lines, rms 0.54px, rated "ok") leans
  1.7 degrees away from the other edges once the photo's angle is taken
  out through the official picture - it was never the card - and is
  dropped (EDGE_LEANING);
- both missing edges are then worked out and looked for again THERE:
  the right edge is found 1.3mm from the box (26 lines, rms ~0.2px) and
  the bottom 7.3mm from it (44 lines, rms ~0.2px) -> EDGE_RELOCATED x2;
- top-to-bottom is MEASURED: shift y ~-0.09mm, T/B ~48/52;
- left-to-right lands 0.01mm under the render-span gate's lower bound
  (0.49 vs 0.50mm) and is refused. The photo is a 9:16 re-encode whose
  quad reads ~1% taller than a card, so that is left alone.
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


def test_leaning_right_edge_dropped_then_found(res):
    r = _edge(res, "right")
    assert r.status in ("ok", "flagged") and r.method != "inferred"
    assert r.n_points >= 15 and r.rms_residual_px <= 1.0
    assert "second look" in " ".join(r.notes)
    assert any(q.code == "EDGE_LEANING" and "right" in q.message
               for q in res.qa)


def test_bottom_relocated_and_measured(res):
    b = _edge(res, "bottom")
    assert b.method in ("step", "texture", "colour")
    assert b.status in ("ok", "flagged")
    assert b.n_points >= 25
    assert b.rms_residual_px <= 1.0
    assert "second look" in " ".join(b.notes)
    codes = [q.code for q in res.qa]
    assert codes.count("EDGE_RELOCATED") == 2
    assert "EDGE_INFERRED" not in codes


def test_top_to_bottom_is_measured(res):
    y = res.shift_mm["y"]
    assert y.status == "measured"
    assert abs(y.value) < 0.35
    assert y.uncertainty.total < 0.3
    assert 40 < res.equivalent_ratio_tb.first_pct < 56


def test_left_to_right_never_passes_silently(res):
    # refused by the span gate today (by 0.01mm); if a future change lets
    # it through it must be a real measurement near the gate, not a guess
    x = res.shift_mm["x"]
    if x.status == "refused":
        assert any(q.code == "RENDER_SPAN_MISMATCH" for q in res.qa)
    else:
        assert x.status == "measured" and x.uncertainty.total < 0.3


# --- Lilo IMG_9467: the cut detector as a last resort ---------------------
# 2160x2880, dark mat, a 6mm glare smear down the right edge, likely foil
# curl. Right and bottom are both refused by the normal detectors. The
# right edge is worked out from the left, and the hybrid cut detector,
# anchored there, reads it from two window widths to within 0.01mm, 0.21mm
# from the prediction -> a "cut-estimate" edge. The bottom's second look
# finds only a scattered texture fit (rms 3.8px, not "ok"), so it is worked
# out instead. The photo quad is then 3.7% off a card's shape and both axes
# are refused by the aspect gate - this capture stays unmeasurable, as the
# failure corpus notes always said. The test pins the mechanics, not a
# number.
LILO = PHOTO.parent / "IMG_9467.jpg"


@pytest.fixture(scope="module")
def lilo():
    if not LILO.exists():
        pytest.skip("failure-corpus photo not present")
    return analyze_borderless(LILO, "13-244", LORCANA, make_overlay=False)


def test_lilo_right_edge_from_the_cut_detector(lilo):
    r = _edge(lilo, "right")
    assert r.method == "cut-estimate" and r.status == "estimated"
    assert r.rms_residual_px <= 1.5 and r.n_points >= 15


def test_lilo_still_refused_honestly(lilo):
    assert lilo.shift_mm["x"].status == "refused"
    assert lilo.shift_mm["y"].status == "refused"
    b = _edge(lilo, "bottom")
    assert b.method == "inferred" or b.status == "refused"
    assert any(q.code == "ASPECT_DEVIATION" for q in lilo.qa)
