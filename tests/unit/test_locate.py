"""Coarse localization on synthetic light backgrounds (polarity support)."""
import numpy as np

from centering.locate import card_component_bbox, coarse_locate

RNG = np.random.default_rng(7)


def synth_scene(W=900, H=700, card=(300, 140, 600, 560), bg=235.0,
                card_level=40.0, artwork=None, noise=2.0):
    img = np.full((H, W), bg, np.float32)
    x0, y0, x1, y1 = card
    img[y0:y1, x0:x1] = card_level
    if artwork:  # bright window inside the card (front artwork)
        ax0, ay0, ax1, ay1, lvl = artwork
        img[ay0:ay1, ax0:ax1] = lvl
    img += RNG.normal(0, noise, (H, W)).astype(np.float32)
    return np.clip(img, 0, 255)


def test_coarse_locate_dark_card_on_white_uses_step_fallback():
    img = synth_scene()
    sides, ppm = coarse_locate(img, 63.5, 88.9)
    for s, true in (("left", 300), ("right", 600), ("top", 140),
                    ("bottom", 560)):
        assert sides[s].status == "ok", (s, sides[s].reason)
        assert sides[s].method == "step"
        assert abs(sides[s].pos - true) < 2.0
    assert ppm is not None and abs(ppm - 300 / 63.5) < 0.2


def test_coarse_locate_still_refuses_zero_contrast():
    img = np.full((700, 900), 200.0, np.float32) \
        + RNG.normal(0, 2.0, (700, 900)).astype(np.float32)
    sides, ppm = coarse_locate(img, 63.5, 88.9)
    assert ppm is None
    for s in ("left", "right", "top", "bottom"):
        assert sides[s].status == "failed"
        assert "contrast" in sides[s].reason


def test_card_component_bbox_dark_card_with_bright_artwork():
    img = synth_scene(artwork=(340, 180, 560, 420, 210.0))
    x0, y0, x1, y1 = card_component_bbox(img)
    assert abs(x0 - 300) <= 6 and abs(x1 - 600) <= 6
    assert abs(y0 - 140) <= 6 and abs(y1 - 560) <= 6


def test_card_component_bbox_bright_card_on_dark_unchanged():
    img = synth_scene(bg=45.0, card_level=190.0)
    x0, y0, x1, y1 = card_component_bbox(img)
    assert abs(x0 - 300) <= 6 and abs(x1 - 600) <= 6
    assert abs(y0 - 140) <= 6 and abs(y1 - 560) <= 6
