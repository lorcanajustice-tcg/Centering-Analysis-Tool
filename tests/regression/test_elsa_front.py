"""Regression: Elsa - Ice Maker 6/C2 borderless front (IMG_6341).

Known-good from the validated prototype analysis: horizontal print shift
0.27mm toward the LEFT card edge (cross-validated against the back at the
0.05mm level). Vertical: originally an open item (no back-side
cross-check); resolved 2026-07-03 by the render-crop bias calibration
(see games/lorcana.py and test_simba_card.py) - the corrected vertical
shift for this card is ~0.0mm and stays inside the loose bound below.

Requires network on first run (render download is cached afterwards).
"""
from pathlib import Path

import pytest

from centering import analyze_borderless, analyze_card
from centering.games.lorcana import LORCANA

ROOT = Path(__file__).resolve().parents[2] / "fixtures"
FRONT = ROOT / "IMG_6341.HEIC"
BACK = ROOT / "IMG_6342.HEIC"

pytestmark = pytest.mark.skipif(not FRONT.exists(),
                                reason="fixture photo not present")


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    out = tmp_path_factory.mktemp("overlays")
    return analyze_borderless(FRONT, "6/C2", LORCANA, out_dir=out)


def test_horizontal_shift_matches_prototype(result):
    m = result.shift_mm["x"]
    assert m.status == "measured"
    # negative = toward left edge; prototype: -0.27, cross-validated
    assert m.value == pytest.approx(-0.27, abs=0.06)
    assert m.uncertainty.total <= 0.06


def test_vertical_shift_measured(result):
    m = result.shift_mm["y"]
    assert m.status == "measured"
    assert abs(m.value) < 0.35  # see module docstring: sign under review


def test_render_match_quality(result):
    r = result.render
    assert r.n_inliers >= 400
    assert r.median_reproj_px <= 1.5
    assert r.render_size == (1468, 2048)


def test_equivalent_ratio(result):
    assert result.equivalent_ratio_lr.display == "45/55"


def test_edge_quality(result):
    ok = {e.edge: e for e in result.edge_fits}
    for side in ("left", "right", "bottom"):
        assert ok[side].rms_residual_px <= 1.0
    assert ok["top"].status in ("ok", "flagged")  # glare band documented


@pytest.mark.skipif(not BACK.exists(), reason="back fixture missing")
def test_combined_registration(tmp_path):
    r = analyze_card(back_photo=BACK, front_photo=FRONT, card_id="6/C2",
                     game=LORCANA, out_dir=tmp_path)
    m = r.registration_mm["x"]
    assert m.status == "measured"
    # prototype: front and back agree at the 0.05mm level
    assert abs(m.value) <= 0.08
    assert r.registration_mm["y"].status == "refused"
    assert r.mirror_consistency
