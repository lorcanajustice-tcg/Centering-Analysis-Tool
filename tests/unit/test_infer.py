"""Inferred fourth edge (infer.py): rebuild one cut edge from the other
three and the card's size, on synthetic quads."""
import numpy as np
import pytest

from centering import geometry as G
from centering import infer as INF

W_MM, H_MM = 62.9, 87.9
IMG = (3000, 4000)


def quad_lines(ppm_top=30.0, ppm_bottom=30.0, x0=500.0, y0=600.0,
               rot=0.0):
    """Four exact lines of a card whose scale changes linearly from the top
    edge to the bottom edge (pitch-style keystone), optionally rotated by a
    small slope `rot`. Returns ({side: FittedLine}, bottom_y_at_centre)."""
    h_px = H_MM * 0.5 * (ppm_top + ppm_bottom)   # linear scale model
    cx = x0 + 0.5 * W_MM * ppm_top
    y1 = y0 + h_px

    def half_w(y):
        t = (y - y0) / h_px
        return 0.5 * W_MM * (ppm_top + t * (ppm_bottom - ppm_top))

    ys = np.linspace(y0, y1, 50)
    left = G.FittedLine.fit("v", ys, cx - half_w(ys) + rot * ys)
    right = G.FittedLine.fit("v", ys, cx + half_w(ys) + rot * ys)
    xs = np.linspace(cx - half_w(y0), cx + half_w(y0), 50)
    top = G.FittedLine.fit("h", xs, y0 - rot * (xs - cx))
    xs_b = np.linspace(cx - half_w(y1), cx + half_w(y1), 50)
    bottom = G.FittedLine.fit("h", xs_b, y1 - rot * (xs_b - cx))
    return {"left": left, "right": right, "top": top, "bottom": bottom}


def _infer(side, lines, **kw):
    three = {k: v for k, v in lines.items() if k != side}
    kw.setdefault("sep_unc_mm", 0.1)
    kw.setdefault("opp_unc_mm", 0.05)
    return INF.infer_missing_edge(side, three, W_MM, H_MM, IMG, **kw)


@pytest.mark.parametrize("side", ["left", "right", "top", "bottom"])
def test_square_on_rebuilds_each_side(side):
    lines = quad_lines()
    inf, why = _infer(side, lines)
    assert inf is not None, why
    u = np.linspace(*inf.line.u_range, 11)
    err_px = np.abs(inf.line.v_at(u) - lines[side].v_at(u))
    assert err_px.max() < 0.05
    assert inf.line.orientation == lines[side].orientation
    assert inf.ppm == pytest.approx(30.0, rel=1e-3)


def test_keystone_follows_linear_scale():
    # bottom 4% closer to the camera than the top
    lines = quad_lines(ppm_top=30.0, ppm_bottom=31.2)
    inf, why = _infer("bottom", lines)
    assert inf is not None, why
    mid = 0.5 * sum(inf.line.u_range)
    assert inf.line.v_at(mid) == pytest.approx(lines["bottom"].v_at(mid),
                                               abs=0.1)
    # and the constructed distance was NOT just card height * top scale
    assert abs(inf.line.v_at(mid) - (600.0 + H_MM * 30.0)) > 20


def test_tilt_term_grows_with_keystone():
    flat, _ = _infer("bottom", quad_lines())
    steep, _ = _infer("bottom", quad_lines(ppm_top=30.0, ppm_bottom=33.0))
    assert flat.tilt_deg == pytest.approx(INF.MIN_TILT_DEG)
    assert steep.tilt_deg > flat.tilt_deg
    assert steep.terms_mm["tilt"] > flat.terms_mm["tilt"]
    assert steep.extra_unc_mm > flat.extra_unc_mm


def test_rotation_is_carried():
    lines = quad_lines(rot=0.02)
    inf, why = _infer("right", lines)
    assert inf is not None, why
    assert inf.line.m == pytest.approx(lines["right"].m, abs=1e-3)


def test_uncertainty_has_every_term_and_a_floor():
    inf, _ = _infer("top", quad_lines(), sep_unc_mm=0.0, opp_unc_mm=0.0)
    assert set(inf.terms_mm) == {"size", "scale", "tilt", "shape",
                                 "opposite_edge"}
    # never better than the manufactured-size spread plus the shape floor
    assert inf.extra_unc_mm >= np.hypot(INF.CUT_SIZE_TOL_MM,
                                        INF.SHAPE_UNC_MM)
    # the perpendicular pair's error is scaled by the aspect ratio
    a, _ = _infer("top", quad_lines(), sep_unc_mm=0.2, opp_unc_mm=0.0)
    assert a.terms_mm["scale"] == pytest.approx(0.2 * H_MM / W_MM)


def test_refuses_when_an_input_is_missing():
    lines = quad_lines()
    three = {k: v for k, v in lines.items() if k not in ("bottom", "left")}
    inf, why = INF.infer_missing_edge("bottom", three, W_MM, H_MM, IMG,
                                      sep_unc_mm=0.1, opp_unc_mm=0.1)
    assert inf is None and "three" in why


def test_refuses_when_construction_leaves_the_photo():
    # top edge missing, and the "bottom" was found on artwork half-way up
    # the card: one card height above it is off the top of the frame
    lines = quad_lines(y0=200.0)
    lines["bottom"] = G.FittedLine.fit(
        "h", np.linspace(600, 2400, 20), np.full(20, 1500.0))
    inf, why = _infer("top", lines)
    assert inf is None and "outside the photo" in why
