"""Inferred fourth edge.

When exactly one cut edge of a front cannot be measured (strict tier and
estimate tier both refused it), the other three still pin it down: the
card is a rectangle of known manufactured size, so the missing edge sits
one card-length away from its opposite edge, in the direction the two
perpendicular edges run.

Why this is worth doing: the front pipeline needs all four edges to build
the quad it hands to render alignment, so one starved edge used to refuse
BOTH axes - including the axis whose two edges were measured cleanly. With
the fourth edge inferred, the clean axis is measured normally and only the
inferred axis is downgraded.

Honesty rules (the project's never-guess ethos):

- the inferred edge is status "inferred", never "ok"; the axis it belongs
  to is at best "estimated" and goes through the estimate tier's cap;
- its uncertainty carries every term the construction introduces: the
  spread of real cut sizes, the error in the pixel scale read off the
  perpendicular pair, camera tilt foreshortening along the missing axis,
  and the opposite edge's own definition (which the inferred edge copies
  one-for-one rather than averaging out);
- it is refused outright when the construction lands outside the photo
  (an input edge must be badly wrong); the caller also refuses it when it
  lands far from where the card was found, and the render-span gate
  downstream checks it against the official picture like any other edge.

Pure geometry; no image access. Scale model: the pixel scale varies
linearly along the missing axis (first-order perspective), read off the
separation of the perpendicular pair.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import geometry as G

# Half-range of manufactured cut sizes. TAG's DIG reports on three Lorcana
# cards span 62.87-62.99 x 87.78-87.94mm (2026-08-21): half-ranges 0.06 and
# 0.08mm. Rounded up to 0.10mm for both, since three cards is a small sample.
CUT_SIZE_TOL_MM = 0.10
# Equivalent focal length assumed when converting keystone into camera
# tilt (a phone main camera). Only feeds an uncertainty term.
ASSUMED_FOCAL_MM_EQUIV = 26.0
# Floor on the tilt used for the foreshortening term: keystone is noisy.
MIN_TILT_DEG = 2.0
# Empirical floor for everything the model above does not capture (the
# measured quad itself is never exactly card-shaped). Leave-one-out over
# the six fixture fronts with four good edges (2026-09-15: drop each edge,
# rebuild it, compare with the measured one; 24 cases): rms error 0.16mm
# excluding the known shadow-band capture (IMG_6330), 0.26mm including it.
# Without this term those 20 clean cases scattered at |z| up to 1.6.
SHAPE_UNC_MM = 0.15

OPPOSITE = {"left": "right", "right": "left", "top": "bottom",
            "bottom": "top"}
_PERP = {"left": ("top", "bottom"), "right": ("top", "bottom"),
         "top": ("left", "right"), "bottom": ("left", "right")}


@dataclass
class InferredEdge:
    line: G.FittedLine
    extra_unc_mm: float
    note: str
    tilt_deg: float
    ppm: float  # local px/mm at the constructed edge
    terms_mm: dict


def infer_missing_edge(side: str, lines: dict, card_w_mm: float,
                       card_h_mm: float, image_size: tuple,
                       sep_unc_mm: float, opp_unc_mm: float,
                       ) -> tuple[Optional[InferredEdge], str]:
    """Build the missing `side` from the other three fitted lines.

    lines        - dict side -> FittedLine for the three measured sides
    sep_unc_mm   - 1-sigma error on the perpendicular pair's separation
    opp_unc_mm   - 1-sigma definition error of the opposite edge
    Returns (InferredEdge, "") or (None, reason).
    """
    opp = lines.get(OPPOSITE[side])
    p1, p2 = (lines.get(s) for s in _PERP[side])
    if opp is None or p1 is None or p2 is None:
        return None, "the other three edges are not all measured"

    vertical = side in ("left", "right")
    # distance to construct, and the perpendicular size that sets the scale
    along_mm, perp_mm = (card_w_mm, card_h_mm) if vertical else \
        (card_h_mm, card_w_mm)
    sign = 1.0 if side in ("right", "bottom") else -1.0

    # u runs along the missing edge; the opposite edge is fitted over u.
    lo, hi = opp.u_range
    if hi - lo < 1.0:
        return None, "the opposite edge is too short to copy"
    us = np.linspace(lo, hi, 41)
    v_opp = opp.v_at(us)

    def perp_sep(w):
        # separation of the perpendicular pair at along-axis position w
        return float(p2.v_at(w) - p1.v_at(w))

    v_new = np.empty_like(v_opp)
    for i, (u, vo) in enumerate(zip(us, v_opp)):
        v = vo + sign * along_mm * perp_sep(vo) / perp_mm
        for _ in range(8):
            # linear scale model: mean scale over [vo, v] = scale at midpoint
            v_next = vo + sign * along_mm * perp_sep(0.5 * (vo + v)) / perp_mm
            if abs(v_next - v) < 1e-4:
                v = v_next
                break
            v = v_next
        v_new[i] = v
    if not np.all(np.isfinite(v_new)):
        return None, "the construction did not converge"

    line = G.FittedLine("v" if vertical else "h",
                        *np.polyfit(us, v_new, 1),
                        rms=opp.rms, n=opp.n, n_rej=0,
                        u_range=(float(lo), float(hi)), bow_px=None)

    # the constructed edge must lie inside the photo (with room for the
    # scanner's profile); otherwise an input edge is badly wrong
    W, H = image_size
    limit = W if vertical else H
    if v_new.min() < 2 or v_new.max() > limit - 3:
        return None, ("working it out from the other three edges puts it "
                      "outside the photo, so one of those edges must be "
                      "wrong")

    # local scale at the new edge, px per mm
    mid_u = 0.5 * (lo + hi)
    v_mid = float(line.v_at(mid_u))
    ppm_here = perp_sep(v_mid) / perp_mm
    if not ppm_here > 0:
        return None, "the edges either side cross over"

    # --- uncertainty terms (mm, 1 sigma) ---
    # size: the constructed distance, plus the perpendicular size that
    # converts px to mm (both are manufactured dimensions)
    size_rel = math.hypot(CUT_SIZE_TOL_MM / along_mm,
                          CUT_SIZE_TOL_MM / perp_mm)
    t_size = along_mm * size_rel
    # scale: error of the perpendicular pair's separation
    t_scale = along_mm * sep_unc_mm / perp_mm
    # tilt: keystone of the perpendicular pair -> tilt about the axis
    # parallel to the missing edge -> foreshortening of the constructed
    # distance by cos(tilt). Sign is known but the focal length is not,
    # so it is carried as uncertainty rather than corrected.
    f_px = ASSUMED_FOCAL_MM_EQUIV / 43.266 * math.hypot(W, H)
    v_opp_mid = float(opp.v_at(mid_u))
    span_px = abs(v_mid - v_opp_mid)
    s_a, s_b = perp_sep(v_opp_mid), perp_sep(v_mid)
    grad = abs(s_b - s_a) / (0.5 * (s_a + s_b)) / max(span_px, 1.0)
    tilt = max(math.degrees(math.atan(f_px * grad)), MIN_TILT_DEG)
    t_tilt = along_mm * (1.0 - math.cos(math.radians(tilt)))
    extra = math.sqrt(t_size ** 2 + t_scale ** 2 + t_tilt ** 2
                      + SHAPE_UNC_MM ** 2 + opp_unc_mm ** 2)
    terms = {"size": t_size, "scale": t_scale, "tilt": t_tilt,
             "shape": SHAPE_UNC_MM, "opposite_edge": opp_unc_mm}
    note = (f"worked out from the {OPPOSITE[side]} edge and the card's "
            f"{along_mm:.1f}mm {'width' if vertical else 'height'}, good to "
            f"about {extra:.2f}mm")
    return InferredEdge(line, extra, note, tilt, ppm_here, terms), ""
