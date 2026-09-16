"""A digital card picture (not a photo) is named as such when it refuses."""
import cv2
import numpy as np

from centering.hexanchor import analyze_corner
from centering.games.lorcana import LORCANA
from centering.imgio import digital_background


def _noisy(shape, level, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(level, 3.0, shape), 0, 255).astype(np.uint8)


def test_padded_digital_picture_is_recognised():
    img = np.zeros((400, 300, 3), np.uint8)
    img[20:380, 20:280] = (120, 90, 60)
    found = digital_background(img)
    assert found is not None
    colour, frac = found
    assert colour == (0, 0, 0) and frac == 1.0


def test_photo_backgrounds_are_not_flagged():
    # sensor noise spreads a background over many values
    assert digital_background(_noisy((400, 300, 3), 40)) is None
    assert digital_background(_noisy((400, 300, 3), 215)) is None
    # an over-exposed white sheet really can clip to pure white
    assert digital_background(np.full((400, 300, 3), 255, np.uint8)) is None
    assert digital_background(np.zeros((40, 30), np.uint8)) is None


def test_refusal_names_the_digital_picture(tmp_path):
    img = np.zeros((500, 400, 3), np.uint8)
    img[30:-30, 30:-30] = (25, 25, 25)      # black border on black
    p = tmp_path / "digital.png"
    cv2.imwrite(str(p), img)
    r = analyze_corner(p, LORCANA, make_overlay=False)
    assert r.shift_mm["x"].status == "refused"
    assert "digital picture" in r.shift_mm["x"].refusal_reason
    assert "printed card" in r.shift_mm["x"].refusal_advice
    assert any(q.code == "DIGITAL_IMAGE" for q in r.qa)
