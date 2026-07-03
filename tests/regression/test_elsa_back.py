"""Regression: Elsa - Ice Maker 6/C2 back photo (IMG_6342).

Known-good from the validated prototype analysis (2026-07-03):
back L/R = 55/45 (L 2.60mm, R 2.16mm), first_pct 54.6 +- ~1.5;
T/B unmeasurable in this shot (glare top, shadow bottom).
"""
from pathlib import Path

import pytest

from centering import analyze_back
from centering.games.lorcana import LORCANA

FIXTURE = Path(__file__).resolve().parents[2] / "fixtures" / "IMG_6342.HEIC"

pytestmark = pytest.mark.skipif(not FIXTURE.exists(),
                                reason="fixture photo not present")


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    out = tmp_path_factory.mktemp("overlays")
    return analyze_back(FIXTURE, LORCANA, out_dir=out)


def test_lr_ratio_matches_prototype(result):
    r = result.ratio_lr
    assert r.status == "measured"
    assert r.first_pct == pytest.approx(54.6, abs=1.5)
    assert r.display in ("54/46", "55/45")


def test_border_widths_match_prototype(result):
    L = result.borders_mm["left"]
    R = result.borders_mm["right"]
    assert L.value == pytest.approx(2.60, abs=0.08)
    assert R.value == pytest.approx(2.16, abs=0.08)
    # +-0.1mm accuracy target: composed uncertainty must stay within it
    assert L.uncertainty.total <= 0.10
    assert R.uncertainty.total <= 0.10


def test_tb_refused_with_reason(result):
    assert result.ratio_tb.status == "refused"
    assert result.ratio_tb.refusal_reason
    for side in ("top", "bottom"):
        m = result.borders_mm[side]
        assert m.status == "refused"
        assert "unmeasurable" in m.refusal_reason or "contrast" in m.refusal_reason


def test_fit_quality(result):
    for e in result.edge_fits:
        if e.status in ("ok", "flagged"):
            assert e.rms_residual_px <= 1.6


def test_deterministic(tmp_path):
    r2 = analyze_back(FIXTURE, LORCANA, out_dir=tmp_path)
    # bit-identical key outputs across runs
    assert r2.ratio_lr.first_pct == analyze_back(
        FIXTURE, LORCANA, out_dir=tmp_path, make_overlay=False).ratio_lr.first_pct


def test_overlay_written(result):
    assert result.overlay and Path(result.overlay).exists()


def test_json_serializable(result):
    import json
    s = json.dumps(result.to_dict())
    assert "55/45" in s or "54/46" in s
