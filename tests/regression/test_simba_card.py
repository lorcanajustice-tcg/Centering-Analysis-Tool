"""Regression: Simba - Pride Protector 8/C2 Enchanted (IMG_6330 front,
IMG_6331 back) - the card pair that exposed the vertical render-crop bias.

History: before calibration the front reported equiv T/B 62/38 while the
back read ~50/50 (implied front-back registration -0.72mm, implausible when
x-registration on the same card is 0.009mm). Root cause: Ravensburger
renders are cropped more off the TOP than the bottom; the old pipeline
assumed a symmetric crop (only ever cross-validated on x). With the
calibrated bias (GameSpec.render_crop_bias_mm) the corrected shift lands
within the shadow-artifact uncertainty of these photos.

Bounds here are deliberately loose: both photos of this pair carry a
shadow band along one horizontal edge (dark textured mat, low sun-angle
light) that the step/texture scanners can lock onto at the ~0.5mm level.

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


def test_vertical_no_longer_wildly_off(result):
    """Pre-fix this read +0.71mm (equiv 62/38). Corrected value must sit
    within the shadow-artifact envelope of these photos."""
    m = result.front.shift_mm["y"]
    assert m.status == "measured"
    assert abs(m.value) < 0.45
    # systematic calibration uncertainty must be carried, not hidden
    assert m.uncertainty.total >= 0.15


def test_equiv_tb_within_envelope(result):
    tb = result.front.equivalent_ratio_tb
    assert tb.status == "measured"
    assert 45.0 <= tb.first_pct <= 60.0  # was 61.9 pre-fix


def test_registration_y_reported(result):
    m = result.registration_mm["y"]
    assert m.status == "measured"
    assert abs(m.value) <= 0.45  # was -0.72 pre-fix
