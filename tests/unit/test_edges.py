import numpy as np

from centering.edges import frame_peak_scan, step_scan, texture_scan
from centering.geometry import FittedLine

RNG = np.random.default_rng(42)


def synth_back(W=900, H=600, edge_x=300.37, mat_sigma=16.0, card_sigma=2.0,
               mat_level=75.0, card_level=70.0):
    img = np.zeros((H, W), np.float32)
    xs = np.arange(W)
    img += RNG.normal(0, card_sigma, (H, W)).astype(np.float32) + card_level
    mat = RNG.normal(0, mat_sigma, (H, W)).astype(np.float32) + mat_level
    m = xs < edge_x
    img[:, m] = mat[:, m]
    return np.clip(img, 0, 255)


def test_texture_scan_subpixel_left_edge():
    edge_x = 300.37
    img = synth_back(edge_x=edge_x)
    us = np.linspace(60, 540, 30)
    u, v, diag = texture_scan(img, "left", 300, us, 200, 120)
    assert diag.n_ok >= 25
    line = FittedLine.fit("v", u, v)
    # texture transition definition is coarse; demand agreement within ~2px
    assert abs(line.v_at(300.0) - edge_x) < 2.5
    assert abs(line.m) < 0.01


def test_texture_scan_refuses_without_contrast():
    img = synth_back(mat_sigma=2.5)  # mat as smooth as the card
    us = np.linspace(60, 540, 20)
    u, v, diag = texture_scan(img, "left", 300, us, 200, 120)
    assert diag.n_ok == 0
    assert "insufficient_texture_contrast" in diag.reject_reasons


def test_step_scan_subpixel_accuracy():
    W, H, edge = 800, 400, 350.6
    xs = np.arange(W, dtype=np.float32)
    ramp = np.clip((xs - (edge - 2.0)) / 4.0, 0, 1)  # 4px linear transition
    row = 25.0 + ramp * 160.0
    img = np.tile(row, (H, 1)) + RNG.normal(0, 2.0, (H, W)).astype(np.float32)
    us = np.linspace(40, 360, 25)
    u, v, diag = step_scan(img, "left", 350, us, 150, 150)
    assert diag.n_ok >= 22
    assert abs(float(np.mean(v)) - edge) < 0.35


def test_step_scan_excludes_low_contrast_lines():
    W, H, edge = 800, 400, 350.0
    xs = np.arange(W, dtype=np.float32)
    ramp = np.clip((xs - edge + 2) / 4.0, 0, 1)
    img = np.tile(25.0 + ramp * 160.0, (H, 1)).astype(np.float32)
    img[150:250] = 60.0  # glare-flattened band: no usable step
    us = np.linspace(60, 340, 20)
    u, v, diag = step_scan(img, "left", 350, us, 150, 150)
    assert diag.reject_reasons.get("insufficient_contrast", 0) >= 5
    assert all(abs(x - edge) < 0.6 for x in v)


def test_frame_peak_prefers_first_peak_from_edge():
    W, H = 800, 400
    img = np.full((H, W), 60.0, np.float32)
    edge_x, frame_x, decoy_x = 100.0, 196.3, 240.0
    xs = np.arange(W, dtype=np.float32)
    for cx, amp in ((frame_x, 70.0), (decoy_x, 90.0)):
        img += amp * np.exp(-0.5 * ((xs - cx) / 2.0) ** 2)[None, :]
    edge = FittedLine("v", 0.0, edge_x, n=30, u_range=(50.0, 350.0))
    us = np.linspace(60, 340, 25)
    ppm = 40.0
    u, v, diag = frame_peak_scan(img, "left", edge, us, ppm)
    assert diag.n_ok >= 24
    assert abs(float(np.mean(v)) - frame_x) < 0.6  # not the brighter decoy
