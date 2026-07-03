import math

import numpy as np
import pytest

from centering.geometry import (FittedLine, corner_angles, corner_quad,
                                intersect, keystone, robust_polyfit,
                                self_calibrate_tilt)


def test_robust_polyfit_recovers_line_with_outliers():
    rng = np.random.default_rng(7)
    u = np.linspace(0, 2000, 50)
    v = 0.004 * u + 100 + rng.normal(0, 0.8, 50)
    v[::7] += 25  # gross outliers
    coeffs, mask, rms = robust_polyfit(u, v)
    assert abs(coeffs[0] - 0.004) < 4e-4
    assert abs(coeffs[1] - 100) < 0.8
    assert rms < 1.2
    assert (~mask).sum() >= 7


def test_robust_polyfit_resists_biased_tail():
    # shadow-drift scenario: last 30% of points biased one way
    rng = np.random.default_rng(3)
    u = np.linspace(0, 2000, 40)
    v = 0.002 * u + 50 + rng.normal(0, 0.7, 40)
    v[-12:] += 4.0  # systematic, same-sign bias clustered at one end
    coeffs, mask, rms = robust_polyfit(u, v)
    assert abs(coeffs[0] - 0.002) < 1.2e-3
    fit_mid = np.polyval(coeffs, 1000)
    assert abs(fit_mid - 52.0) < 1.6


def test_intersect_perpendicular():
    vline = FittedLine("v", 0.0, 100.0)   # x = 100
    hline = FittedLine("h", 0.0, 200.0)   # y = 200
    assert intersect(vline, hline) == (100.0, 200.0)


def test_corner_quad_and_keystone_rectangle():
    l = FittedLine("v", 0.0, 0.0)
    r = FittedLine("v", 0.0, 630.0)
    t = FittedLine("h", 0.0, 0.0)
    b = FittedLine("h", 0.0, 880.0)
    q = corner_quad(l, r, t, b)
    np.testing.assert_allclose(q, [[0, 0], [630, 0], [630, 880], [0, 880]])
    kw, kh = keystone(q)
    assert kw == pytest.approx(0.0)
    assert kh == pytest.approx(0.0)
    ang = corner_angles(q)
    for v in ang.values():
        assert v == pytest.approx(90.0)


def _project(f, W, H, pitch_deg, yaw_deg, card_w, card_h, z=350.0):
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    rx, ry = math.radians(pitch_deg), math.radians(yaw_deg)
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(rx), -math.sin(rx)],
                   [0, math.sin(rx), math.cos(rx)]])
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)],
                   [0, 1, 0],
                   [-math.sin(ry), 0, math.cos(ry)]])
    R = Rx @ Ry
    corners = np.array([[0, 0, 0], [card_w, 0, 0],
                        [card_w, card_h, 0], [0, card_h, 0]], float)
    corners -= [card_w / 2, card_h / 2, 0]
    cam = (R @ corners.T).T + [0, 0, z]
    px = (K @ cam.T).T
    return px[:, :2] / px[:, 2:3]


def test_self_calibrate_tilt_recovers_angles():
    f, W, H = 9000.0, 3024, 4032
    quad = _project(f, W, H, pitch_deg=2.0, yaw_deg=1.5,
                    card_w=63.5, card_h=88.9)
    out = self_calibrate_tilt(quad, 63.5, 88.9, (W, H))
    assert out is not None
    f_px, f_eq, pitch, yaw, total = out
    assert f_px == pytest.approx(f, rel=0.08)
    expected_total = math.degrees(math.acos(
        math.cos(math.radians(2.0)) * math.cos(math.radians(1.5))))
    assert total == pytest.approx(expected_total, abs=0.35)


def test_self_calibrate_degenerate_frontoparallel():
    f, W, H = 9000.0, 3024, 4032
    quad = _project(f, W, H, pitch_deg=0.0, yaw_deg=0.0,
                    card_w=63.5, card_h=88.9)
    out = self_calibrate_tilt(quad, 63.5, 88.9, (W, H))
    # fronto-parallel: either honestly degenerate or ~zero tilt
    if out is not None:
        assert out[4] < 0.6
