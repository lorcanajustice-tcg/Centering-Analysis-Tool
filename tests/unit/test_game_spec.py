"""GameSpec edge-definition composition (edge_def_px + cut_def_mm).

The two terms are different physics and the arithmetic below is what keeps
them from being confused: `edge_def_px` is a DETECTOR property and shrinks
in mm as resolution rises; `cut_def_mm` is a property of the printed FACE
and does not. See games/lorcana.py for the Lorcana provenance and
DEV-NOTES.md ("Acting on the DIG dimensions").
"""
import math

import pytest

from centering.games.base import GameSpec
from centering.games.lorcana import LORCANA


def _spec(**kw):
    return GameSpec(name="t", card_w_mm=62.9, card_h_mm=87.9, **kw)


def test_without_a_face_term_it_is_the_old_pixel_arithmetic():
    """A game that sets no cut_def_mm must be bit-identical to the
    pre-2026-08-21 behaviour (plain edge_def_px / px_per_mm)."""
    g = _spec()
    for method in ("texture", "step", "colour", "frame_peak"):
        for ppm in (30.0, 68.0):
            for axis in ("x", "y"):
                assert g.edge_def_mm("front", method, axis, ppm) == \
                    pytest.approx(g.edge_def_px[method] / ppm)


def test_face_term_is_added_in_quadrature_per_axis():
    g = _spec(cut_def_mm={"front": {"x": 0.05, "y": 0.08}})
    assert g.edge_def_mm("front", "step", "x", 68.0) == pytest.approx(
        math.hypot(1.0 / 68.0, 0.05))
    assert g.edge_def_mm("front", "step", "y", 68.0) == pytest.approx(
        math.hypot(1.0 / 68.0, 0.08))
    # a face that is not listed carries none of it
    assert g.edge_def_mm("back", "step", "x", 68.0) == pytest.approx(1.0 / 68.0)


def test_the_two_terms_scale_differently_with_resolution():
    """The point of keeping them separate: 4x the resolution quarters the
    detector term and leaves the physical one alone."""
    g = _spec(cut_def_mm={"front": {"x": 0.05}})
    lo = g.edge_def_mm("front", "texture", "x", 17.0)   # 3px = 0.176mm
    hi = g.edge_def_mm("front", "texture", "x", 68.0)   # 3px = 0.044mm
    assert lo > hi
    assert hi == pytest.approx(math.hypot(3.0 / 68.0, 0.05))
    # however fine the capture, the face term is the floor
    assert g.edge_def_mm("front", "texture", "x", 10_000.0) == \
        pytest.approx(0.05, abs=1e-4)


def test_unknown_detector_falls_back_to_step():
    g = _spec()
    assert g.edge_def_mm("back", "no_such_scanner", "x", 50.0) == \
        pytest.approx(g.edge_def_px["step"] / 50.0)


def test_lorcana_front_is_less_certain_than_its_back():
    """The DIG dimensions: the back's cut is right to <0.01mm while the
    same card's front reads 0.05mm/side wider in x, 0.08 in y."""
    for axis, expected in (("x", 0.05), ("y", 0.08)):
        back = LORCANA.edge_def_mm("back", "colour", axis, 68.0)
        front = LORCANA.edge_def_mm("front", "colour", axis, 68.0)
        assert front > back
        assert front == pytest.approx(math.hypot(back, expected))
