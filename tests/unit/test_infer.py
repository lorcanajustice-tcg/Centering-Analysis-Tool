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
    assert inf.scale_from == "pair"
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
    assert inf is None and "too few" in why
    # ...and with no opposite edge at all
    two = {k: v for k, v in lines.items() if k not in ("bottom", "top")}
    inf, why = INF.infer_missing_edge("bottom", two, W_MM, H_MM, IMG,
                                      sep_unc_mm=0.1, opp_unc_mm=0.1)
    assert inf is None and "opposite" in why


def test_refuses_when_construction_leaves_the_photo():
    # top edge missing, and the "bottom" was found on artwork half-way up
    # the card: one card height above it is off the top of the frame
    lines = quad_lines(y0=200.0)
    lines["bottom"] = G.FittedLine.fit(
        "h", np.linspace(600, 2400, 20), np.full(20, 1500.0))
    inf, why = _infer("top", lines)
    assert inf is None and "outside the photo" in why


# --- rectified construction (with the photo -> picture alignment) ------

def _pitched_scene(tilt_deg=12.0, f=3000.0):
    """A card plane viewed with strong pitch: returns (lines, H) where H
    maps photo px to a 10 px/mm card-plane frame (origin at the card's
    top-left). Linear-scale models cannot reproduce this."""
    import cv2
    t = np.radians(tilt_deg)
    R = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)],
                  [0, np.sin(t), np.cos(t)]])
    K = np.array([[f, 0, 1500], [0, f, 2000], [0, 0, 1]])
    corners_mm = np.array([[-W_MM / 2, -H_MM / 2], [W_MM / 2, -H_MM / 2],
                           [W_MM / 2, H_MM / 2], [-W_MM / 2, H_MM / 2]])
    P = []
    for x, y in corners_mm:
        X = R @ np.array([x, y, 0.0]) + np.array([0, 0, 150.0])
        p = K @ X
        P.append(p[:2] / p[2])
    P = np.array(P, np.float32)
    frame = np.array([[0, 0], [W_MM * 10, 0], [W_MM * 10, H_MM * 10],
                      [0, H_MM * 10]], np.float32)
    H = cv2.getPerspectiveTransform(P, frame)

    def line(a, b, orient):
        t_ = np.linspace(0, 1, 40)[:, None]
        pts = P[a] + t_ * (P[b] - P[a])
        if orient == "v":
            return G.FittedLine.fit("v", pts[:, 1], pts[:, 0])
        return G.FittedLine.fit("h", pts[:, 0], pts[:, 1])
    lines = {"top": line(0, 1, "h"), "right": line(1, 2, "v"),
             "bottom": line(3, 2, "h"), "left": line(0, 3, "v")}
    return lines, H.astype(np.float64)


@pytest.mark.parametrize("side", ["left", "right", "top", "bottom"])
def test_rectified_is_exact_under_strong_pitch(side):
    lines, H = _pitched_scene()
    rect, why = _infer(side, lines, H_photo_to_render=H)
    assert rect is not None, why
    u = np.linspace(*rect.line.u_range, 11)
    assert np.abs(rect.line.v_at(u) - lines[side].v_at(u)).max() < 0.5
    assert "alignment" in rect.terms_mm and "tilt" not in rect.terms_mm


def test_linear_model_misses_pitch_along_the_edge():
    # the case the rectified path exists for: a left/right edge under
    # pitch leans, and the linear model copies the opposite edge's lean
    lines, H = _pitched_scene()
    lin, _ = _infer("right", lines)
    rect, _ = _infer("right", lines, H_photo_to_render=H)
    true_m = lines["right"].m
    assert abs(rect.line.m - true_m) < 2e-3
    assert abs(lin.line.m - true_m) > 5 * abs(rect.line.m - true_m)


def test_picture_scale_used_without_the_pair():
    lines, H = _pitched_scene()
    only = {"top": lines["top"]}          # bottom missing, no left/right
    inf, why = INF.infer_missing_edge(
        "bottom", only, W_MM, H_MM, IMG, sep_unc_mm=0.0, opp_unc_mm=0.05,
        H_photo_to_render=H, frame_ppm=10.0, frame_ppm_rel_unc=0.005)
    assert inf is not None, why
    assert inf.scale_from == "picture"
    assert "official picture" in inf.note
    assert inf.terms_mm["scale"] == pytest.approx(H_MM * 0.005)
    u = np.linspace(*inf.line.u_range, 11)
    assert np.abs(inf.line.v_at(u) - lines["bottom"].v_at(u)).max() < 0.5
    # a wrong picture scale moves it by exactly that fraction of a card
    off, _ = INF.infer_missing_edge(
        "bottom", only, W_MM, H_MM, IMG, sep_unc_mm=0.0, opp_unc_mm=0.05,
        H_photo_to_render=H, frame_ppm=10.1, frame_ppm_rel_unc=0.005)
    mid = 0.5 * sum(inf.line.u_range)
    shift_mm = (off.line.v_at(mid) - inf.line.v_at(mid)) / inf.ppm
    assert shift_mm == pytest.approx(0.01 * H_MM, rel=0.1)


def test_picture_scale_needs_the_alignment():
    lines, _ = _pitched_scene()
    inf, why = INF.infer_missing_edge(
        "bottom", {"top": lines["top"]}, W_MM, H_MM, IMG, sep_unc_mm=0.0,
        opp_unc_mm=0.05, frame_ppm=10.0)
    assert inf is None and "too few" in why
