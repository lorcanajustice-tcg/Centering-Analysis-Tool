"""Regression: Simba - Pride Protector 8/C2 Enchanted (IMG_6330 front,
IMG_6331 back) - the card pair that exposed the vertical render-crop bias.

History: this pair exposed that the old pipeline assumed a symmetric
render crop (front read equiv T/B 62/38 vs back ~50/50, implied
registration -0.72mm). The 2026-07-03 bias (+0.20) made it "land within
the envelope" - but that constant was itself derived from the same
shadow-contaminated dark-mat shoot. The 2026-07-06 recalibration under
the white-paper protocol (bias y -0.08, consistent with zero) removes the
masking.

Both photos carry a shadow band along one horizontal edge (dark textured
mat, low sun-angle light). 2026-07-19 finding: the front edge fit on this
capture is ENVIRONMENT-SENSITIVE - the back face reproduces to 4 decimals
across machines, but whether the front top-edge fit latches onto the
shadow band's outer boundary varies with the OpenCV stack:

* cv2 4.13 / py3.10: fit dodges the shadow -> spans inside the
  render-span gate -> everything MEASURED, and the ~0.5mm artifact shows
  up honestly as the front-back registration y disagreement.
* cv2 5.0 / py3.14: top edge drags +0.5mm onto the shadow -> the
  d3b0f9f render-span plausibility gate fires (y span 2.37 vs
  1.05..2.15mm, x 1.27 vs 0.50..1.15mm) -> the front axes are REFUSED
  with RENDER_SPAN_MISMATCH.

BOTH are honest outcomes for a knowingly compromised capture: either the
artifact is measured and exposed, or the contaminated fit is refused
outright. What this file regression-tests is that the artifact is never
silently MASKED (a clean-looking, small-disagreement "measured" result).
The white-background protocol fixtures (test_gadget_multibg) are
env-stable; only this dark-mat shadow capture is bimodal.

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


def _front_y_refused_by_span_gate(result):
    m = result.front.shift_mm["y"]
    return (m.status == "refused"
            and "render-to-cut" in (m.refusal_reason or ""))


def test_back_face_is_env_stable(result):
    """The back scan reproduced to 4 decimals across cv2 4.13/5.0."""
    b = result.back.borders_mm
    for side in ("left", "right", "top", "bottom"):
        assert b[side].status == "measured"
    assert 2.0 <= b["left"].value <= 2.4      # 2.186 observed
    assert 2.3 <= b["right"].value <= 2.7     # 2.487
    assert 2.0 <= b["top"].value <= 2.35      # 2.161
    assert 2.0 <= b["bottom"].value <= 2.35   # 2.144


def test_bias_correction_applied(result):
    codes = [q.code for q in result.front.qa]
    assert "RENDER_CROP_BIAS_CORRECTED" in codes


def test_refusal_mode_is_flagged(result):
    """If the span gate fired it must say so in QA, and the refusal reason
    must name the physical implausibility - never a silent None."""
    if _front_y_refused_by_span_gate(result):
        codes = [q.code for q in result.front.qa]
        assert "RENDER_SPAN_MISMATCH" in codes
        assert "layout-locked" in result.front.shift_mm["y"].refusal_reason


def test_x_axis_cross_validation(result):
    """Measured mode: front-back x registration stays tight (0.009 at
    introduction, -0.071 on cv2 4.13). Refused mode: the refusal must
    trace back to the span gate, not something new."""
    m = result.registration_mm["x"]
    if m.status == "measured":
        assert abs(m.value) <= 0.15
    else:
        fx = result.front.shift_mm["x"]
        assert fx.status == "refused"
        assert "render-to-cut" in (fx.refusal_reason or "")


def test_vertical_artifact_never_masked(result):
    """The shadow artifact must surface: either the front y shift is
    measured at ~+0.5mm (0.52 at introduction) with the systematic
    uncertainty carried, or the contaminated fit is refused by the span
    gate. A small, clean-looking y shift would mean masking - fail."""
    m = result.front.shift_mm["y"]
    if m.status == "measured":
        assert 0.2 <= m.value <= 0.8
        assert m.uncertainty.total >= 0.10
    else:
        assert _front_y_refused_by_span_gate(result)


def test_equiv_tb_follows_y_mode(result):
    tb = result.front.equivalent_ratio_tb
    m = result.front.shift_mm["y"]
    if m.status == "measured":
        assert tb.status == "measured"
        assert 55.0 <= tb.first_pct <= 70.0   # 62.1 at introduction
    else:
        assert tb.status == "refused"


def test_registration_y_exposes_or_refuses(result):
    """Measured mode: the front-back y disagreement IS the shadow artifact
    (-0.53 at introduction; the old +0.20 constant made it read -0.25).
    Refused mode: the registration refusal must cite the front y refusal."""
    m = result.registration_mm["y"]
    if m.status == "measured":
        assert 0.3 <= abs(m.value) <= 0.8
    else:
        assert _front_y_refused_by_span_gate(result)
        assert "front y" in (m.refusal_reason or "")
