import json

from centering.types import Measurement, Ratio, Uncertainty
from centering.uncertainty import ratio_pts_sigma


def test_uncertainty_total_rss():
    u = Uncertainty(3.0, 4.0, 0.0)
    assert u.total == 5.0


def test_measurement_serializes():
    m = Measurement(2.6026, "mm", Uncertainty(0.01, 0.02, 0.08))
    d = m.to_dict()
    json.dumps(d)
    assert d["status"] == "measured"
    assert d["uncertainty"]["total"] > 0.08


def test_refused_measurement():
    m = Measurement.refused("mm", "top edge unmeasurable: insufficient contrast")
    d = m.to_dict()
    assert d["value"] is None
    assert d["status"] == "refused"
    assert "unmeasurable" in d["refusal_reason"]


def test_ratio_display_and_convention():
    r = Ratio(axis="LR", first_pct=54.64)
    assert r.display == "55/45"
    d = r.to_dict()
    assert d["second_pct"] == 45.4  # 0.1-precision carried alongside
    assert "Left" in d["convention"]


def test_ratio_pts_sigma_symmetric():
    # 50/50 with equal border sigmas: sigma_pts = 100*sigma/(sqrt(2)*S)
    s = ratio_pts_sigma(2.0, 2.0, 0.05, 0.05)
    assert abs(s - 100 * 0.05 * (8 ** 0.5) / 16.0) < 1e-9
