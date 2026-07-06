"""Regression: Simba - Pride Protector 8/C2 Enchanted (IMG_6330 front,
IMG_6331 back) - the card pair that exposed the vertical render-crop bias.

History: this pair exposed that the old pipeline assumed a symmetric
render crop (front read equiv T/B 62/38 vs back ~50/50, implied
registration -0.72mm). The 2026-07-03 bias (+0.20) made it "land within
the envelope" - but that constant was itself derived from the same
shadow-contaminated dark-mat shoot. The 2026-07-06 recalibration under
the white-paper protocol (bias y -0.08, consistent with zero) removes the
masking: this pair now HONESTLY shows its ~0.5mm shadow artifact as a
front-vs-back y disagreement. That is the correct behaviour for a
compromised capture, and what this file now regression-tests.

Both photos carry a shadow band along one horizontal edge (dark textured
mat, low sun-angle light). The per-face scans are internally
self-consistent (the 2026-07-06 hybrid cross-check reads only ~0.1mm
displacement per face), so the FRONT-BACK REGISTRATION cross-check is the
mechanism that catches this artifact - not a per-edge QA flag.

Requires network on first run (render download is cached afterwards).
"""
from pathlib import Path

import pytest

from centering import analyze_card
from centering.games.lorcana import LORCANA

ROOT = Path(__file__).resolve().parents[2] / "fixtures"
FRONT = ROOT / "IMG_6330.HEIC"
BACK = ROOT / "IMG_6331.HEIC"

pytestmark = pytest.mark.skipif(not (FRONT.exists() and BACK.exists()),
                                reason="fixture photos not present")


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    out = tmp_path_factory.mktemp("overlays")
    return analyze_card(back_photo=BACK, front_photo=FRONT, card_id="8/C2",
                        game=LORCANA, out_dir=out)


def test_bias_correction_applied(result):
    codes = [q.code for q in result.front.qa]
    assert "RENDER_CROP_BIAS_CORRECTED" in codes


def test_x_axis_cross_validation_still_tight(result):
    m = result.registration_mm["x"]
    assert m.status == "measured"
    assert abs(m.value) <= 0.10  # historically 0.009


def test_vertical_shows_shadow_artifact(result):
    """~+0.5mm on the front face (shadow-displaced top edge); at
    introduction of the honest bias: +0.52."""
    m = result.front.shift_mm["y"]
    assert m.status == "measured"
    assert 0.2 <= m.value <= 0.8
    # systematic calibration uncertainty must be carried, not hidden
    assert m.uncertainty.total >= 0.10


def test_equiv_tb_within_envelope(result):
    tb = result.front.equivalent_ratio_tb
    assert tb.status == "measured"
    assert 55.0 <= tb.first_pct <= 70.0  # 62.1 at introduction


def test_registration_y_exposes_artifact(result):
    """The front-back y disagreement IS the shadow artifact; the honest
    bias must not mask it (-0.53 at introduction; the old +0.20 constant
    made it read -0.25)."""
    m = result.registration_mm["y"]
    assert m.status == "measured"
    assert 0.3 <= abs(m.value) <= 0.8
