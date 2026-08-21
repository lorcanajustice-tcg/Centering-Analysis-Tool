"""Estimate-tier edge rescue (estimate.py): cluster split, hybrid
arbitration, and honest refusals, on synthetic imagery."""
import numpy as np

from centering import estimate as ET

PPM = 10.0  # synthetic scale: 10 px per mm


def synth_card(edge_x=150.0, W=800, H=1000, bg=30.0, card=180.0):
    """Bright card occupying x >= edge_x on a dark mat (left edge case)."""
    g = np.full((H, W), bg, dtype=np.float32)
    g[:, int(edge_x):] = card
    return g


def det_cluster(v0, n=12, u_lo=300, u_hi=700, jitter=0.3, seed=0):
    rng = np.random.default_rng(seed)
    u = np.linspace(u_lo, u_hi, n)
    v = v0 + rng.normal(0.0, jitter, n)
    return u, v


US = np.linspace(300, 700, 50)


def test_single_clean_cluster_rescued():
    g = synth_card(150.0)
    u, v = det_cluster(150.0)
    est, why = ET.rescue_edge(g, "left", u, v, US, PPM)
    assert est is not None, why
    assert abs(est.line.v_at(500.0) - 150.0) < 1.0
    assert ET.BASE_SYSTEMATIC_MM <= est.extra_unc_mm <= ET.CAP_MM


def test_sleeve_contaminant_cluster_arbitrated_away():
    # second consistent cluster 1.2mm OUTSIDE the true cut (sleeve edge /
    # glare boundary signature); the hybrid cross-check must pick the cut
    g = synth_card(150.0)
    u1, v1 = det_cluster(150.0, n=10, seed=1)
    u2, v2 = det_cluster(138.0, n=8, u_lo=320, u_hi=680, seed=2)
    est, why = ET.rescue_edge(g, "left", np.concatenate([u1, u2]),
                              np.concatenate([v1, v2]), US, PPM)
    assert est is not None, why
    assert abs(est.line.v_at(500.0) - 150.0) < 1.0


def test_too_few_points_refused():
    g = synth_card(150.0)
    est, why = ET.rescue_edge(g, "left", np.array([400.0, 500.0]),
                              np.array([150.0, 150.2]), US, PPM)
    assert est is None
    assert "usable readings" in why


def test_short_span_refused():
    g = synth_card(150.0)
    u, v = det_cluster(150.0, n=8, u_lo=480, u_hi=520)
    est, why = ET.rescue_edge(g, "left", u, v, US, PPM)
    assert est is None


def test_no_image_support_refused():
    # detections form a perfectly consistent line, but the image holds no
    # edge there -- the hybrid cross-check must veto the rescue
    g = synth_card(400.0)
    u, v = det_cluster(150.0)
    est, why = ET.rescue_edge(g, "left", u, v, US, PPM)
    assert est is None
    assert "second, independent check" in why


def test_scattered_detections_refused():
    # uniform 6mm scatter with NO supporting edge anywhere near it: even
    # if a chance sub-cluster is internally consistent, the hybrid
    # cross-check finds no cut there and the rescue must refuse. (With a
    # real edge under the scatter a chance cluster ON the edge may be
    # legitimately rescued -- that case is covered by the tests above.)
    g = synth_card(400.0)
    rng = np.random.default_rng(3)
    u = np.linspace(300, 700, 16)
    v = 150.0 + rng.uniform(-30, 30, 16)
    est, why = ET.rescue_edge(g, "left", u, v, US, PPM)
    assert est is None
