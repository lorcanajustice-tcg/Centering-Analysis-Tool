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
    assert "top" in v["y"] and "together sit" in v["y"]


def test_cut_inside_render_is_impossible():
    v = _render_span_violations(_off(-0.30, 0.90, 0.70, 0.90), BOUNDS)
    assert "x" in v and "left" in v["x"]


def test_compensating_displacements_caught_per_side():
    # both y edges displaced outward/inward so the total stays legal:
    # the per-side bound must still fire
    v = _render_span_violations(_off(0.30, 0.50, 2.00, -0.05), BOUNDS)
    assert "y" in v


# --- leaning-edge check -------------------------------------------------

def _lines_rotated(deg, lean=None):
    """Card-shaped lines rotated by `deg`; `lean` = (side, extra_deg)."""
    import math
    import numpy as np
    from centering import geometry as G
    m = math.tan(math.radians(deg))
    out = {}
    for side, (o, b) in {"left": ("v", 100.0), "right": ("v", 729.0),
                         "top": ("h", 100.0), "bottom": ("h", 979.0)}.items():
        mm = m if o == "h" else -m
        if lean and lean[0] == side:
            extra = math.tan(math.radians(lean[1]))
            mm = mm + (extra if o == "h" else -extra)
        u = np.linspace(150, 850, 30)
        out[side] = G.FittedLine.fit(o, u, mm * u + b)
    return out


def test_lean_check_accepts_a_rotated_card():
    import numpy as np
    from centering.borderless import _lean_check
    assert _lean_check(np.eye(3), _lines_rotated(0.8), {}) == {}


def test_lean_check_names_the_odd_edge_out():
    import numpy as np
    from centering.borderless import LEAN_MAX_DEG, _lean_check
    got = _lean_check(np.eye(3),
                      _lines_rotated(0.3, ("right", LEAN_MAX_DEG + 0.5)), {})
    assert list(got) == ["right"]
    assert "degrees" in got["right"]
    # just under the limit is fine
    assert _lean_check(np.eye(3),
                       _lines_rotated(0.3, ("right", LEAN_MAX_DEG - 0.2)),
                       {}) == {}


def test_lean_check_skips_inferred_edges_and_needs_three():
    import numpy as np
    from centering.borderless import _lean_check
    lines = _lines_rotated(0.0, ("top", 5.0))
    assert _lean_check(np.eye(3), lines, {"top": "inferred"}) == {}
    lines["left"] = None
    assert _lean_check(np.eye(3), lines, {"top": "inferred"}) == {}


def test_span_gate_slack_for_estimated_edges():
    base = _off(0.288, 0.44, 0.638, 1.329)
    wide = dict(base, right_outside_render=1.10)     # x total 1.388
    assert "x" in _render_span_violations(wide, BOUNDS)
    assert "x" not in _render_span_violations(wide, BOUNDS,
                                              {"right": 0.3})
