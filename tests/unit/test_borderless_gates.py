"""Render-span physical-plausibility gate (borderless fronts).

Numbers below are real measurements from 2026-07-06: the clean Woody
54/P3 capture and the cast-shadow-contaminated Tramp 7:212 capture
(IMG_6416), whose top edge scan latched onto the card's own hard cast
shadow (sharp, tonally continuous with the dark art) 1.4mm outside the
true cut - invisible to per-line edge QA, caught only by the span gate.
"""
from centering.borderless import _render_span_violations

BOUNDS = {"x_total": (0.50, 1.15), "y_total": (1.05, 2.15),
          "side": (-0.10, 1.90)}


def _off(left, right, top, bottom):
    return {"left_outside_render": left, "right_outside_render": right,
            "top_outside_render": top, "bottom_outside_render": bottom}


def test_clean_capture_passes():
    # Woody 54/P3, clean white-paper capture
    assert _render_span_violations(_off(0.288, 0.44, 0.638, 1.329),
                                   BOUNDS) == {}


def test_cast_shadow_top_refuses_y_only():
    # Tramp 7:212 (IMG_6416): top edge on the cast shadow, x axis clean
    v = _render_span_violations(_off(0.295, 0.543, 2.238, 0.53), BOUNDS)
    assert set(v) == {"y"}
    assert "top" in v["y"] and "span" in v["y"]


def test_cut_inside_render_is_impossible():
    v = _render_span_violations(_off(-0.30, 0.90, 0.70, 0.90), BOUNDS)
    assert "x" in v and "left" in v["x"]


def test_compensating_displacements_caught_per_side():
    # both y edges displaced outward/inward so the total stays legal:
    # the per-side bound must still fire
    v = _render_span_violations(_off(0.30, 0.50, 2.00, -0.05), BOUNDS)
    assert "y" in v
