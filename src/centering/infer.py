"""Inferred missing edge.

When a cut edge of a front cannot be measured (strict tier and estimate
tier both refused it, or it was dropped for leaning), the card being a
rectangle of known manufactured size still pins it down: it sits one
card-length from its opposite edge, parallel to it.

Why this is worth doing: the front pipeline needs all four edges to build
the quad it hands to render alignment, so one starved edge used to refuse
BOTH axes - including the axis whose two edges were measured cleanly. With
the edge inferred, the clean axis is measured normally and only the
inferred axis is downgraded. The inferred position is also where the
caller looks for the real edge a second time.

Two ways to get the scale (px per mm) that turns a card-length into
pixels:

- "pair": the separation of the two perpendicular edges, divided by the
  card's other dimension. Needs both of them measured.
- "picture": the official picture's own calibrated size (its crop inside
  the cut is layout-locked, see GameSpec.render_span_bounds_mm). Needs the
  alignment, and is looser (~0.5%), so it is used only when the
  perpendicular pair is not available.

With the alignment (photo -> official picture homography) the
construction is done in the picture's frame, where perspective is already
undone, and mapped back: exact under any camera tilt. Without it a
linear-scale model is used, which cannot see tilt along the missing axis,
so it carries a tilt term instead.

Honesty rules (the project's never-guess ethos):

- the inferred edge is method "inferred", status "estimated"; its axis is
  at best "estimated" and goes through the estimate tier's cap;
- its uncertainty carries every term the construction introduces: the
  spread of real cut sizes, the scale error, tilt (or alignment) error, an
  empirical shape floor, and the opposite edge's own definition (which the
  inferred edge copies one-for-one rather than averaging out);
- it is refused outright when the construction lands outside the photo
  (an input edge must be badly wrong); the caller also refuses it when it
  lands far from where the card was found, and the render-span gate
  downstream checks it (with its margin) like any other edge.

Pure geometry; no image access.
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
# tilt (a phone main camera). Only feeds an uncertainty term, and only in
# the linear (no-alignment) model.
ASSUMED_FOCAL_MM_EQUIV = 26.0
# Floor on the tilt used for the foreshortening term: keystone is noisy.
MIN_TILT_DEG = 2.0
# Empirical floor for everything the models do not capture (the measured
# quad itself is never exactly card-shaped). Leave-one-out over the six
# fixture fronts with four good edges (2026-09-15: drop each edge, rebuild
# it, compare with the measured one; 24 cases per mode), rms error:
#   linear model, pair scale      0.16mm (0.26 incl. IMG_6330)
#   rectified,    pair scale      0.10mm (0.18 incl. IMG_6330), |z|<=1.81
#   rectified,    picture scale   0.24mm (0.23 incl.),          |z|<=0.78
# IMG_6330 is the known shadow-band capture: its measured edges are the
# ones that are off. Kept at the linear model's level, i.e. conservative
# for the rectified path the pipeline normally takes.
SHAPE_UNC_MM = 0.15

OPPOSITE = {"left": "right", "right": "left", "top": "bottom",
            "bottom": "top"}
PERP = {"left": ("top", "bottom"), "right": ("top", "bottom"),
        "top": ("left", "right"), "bottom": ("left", "right")}


@dataclass
class InferredEdge:
    line: G.FittedLine
    extra_unc_mm: float
    note: str
    tilt_deg: float          # linear model only; 0 when rectified
    ppm: float               # local photo px/mm at the constructed edge
    terms_mm: dict
    scale_from: str          # "pair" | "picture"


def _fit_in(H, line, n=60):
    """Slope/intercept of `line` (photo px) re-fitted in H's frame, same
    orientation convention."""
    pts = G.transform_points(H, line.points(n))
    if line.orientation == "v":
        return np.polyfit(pts[:, 1], pts[:, 0], 1)
    return np.polyfit(pts[:, 0], pts[:, 1], 1)


def _mid_point(line):
    u = 0.5 * sum(line.u_range)
    v = float(line.v_at(u))
    return (v, u) if line.orientation == "v" else (u, v)


def infer_missing_edge(side: str, lines: dict, card_w_mm: float,
                       card_h_mm: float, image_size: tuple,
                       sep_unc_mm: float, opp_unc_mm: float,
                       H_photo_to_render: Optional[np.ndarray] = None,
                       h_unc_mm: float = 0.0,
                       frame_ppm: Optional[float] = None,
                       frame_ppm_rel_unc: float = 0.0,
                       ) -> tuple[Optional[InferredEdge], str]:
    """Build the missing `side` from its opposite edge and a scale.

    lines        - dict side -> FittedLine (None / absent = not usable)
    sep_unc_mm   - 1-sigma error on the perpendicular pair's separation
    opp_unc_mm   - 1-sigma definition error of the opposite edge
    H_photo_to_render - photo -> official picture homography, or None
    h_unc_mm     - 1-sigma position error the alignment itself adds
    frame_ppm    - picture px per mm along the missing axis ("picture"
                   scale); used only when the perpendicular pair is absent
    frame_ppm_rel_unc - its 1-sigma relative error
    Returns (InferredEdge, "") or (None, reason).
    """
    opp = lines.get(OPPOSITE[side])
    p1, p2 = (lines.get(s) for s in PERP[side])
    H = H_photo_to_render
    if opp is None:
        return None, "the edge opposite it was not measured either"
    pair = p1 is not None and p2 is not None
    if not pair and (H is None or not frame_ppm):
        return None, ("too few of the other edges were measured to work "
                      "out where it should be")

    vertical = side in ("left", "right")
    along_mm, perp_mm = (card_w_mm, card_h_mm) if vertical else \
        (card_h_mm, card_w_mm)
    sign = 1.0 if side in ("right", "bottom") else -1.0
    lo, hi = opp.u_range
    if hi - lo < 1.0:
        return None, "the opposite edge is too short to copy"
    us = np.linspace(lo, hi, 41)
    W, Himg = image_size

    tilt = 0.0
    if H is not None:
        try:
            Hinv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return None, "the alignment with the official picture is broken"
        mo, bo = _fit_in(H, opp)
        ox, oy = G.transform_points(H, np.array([_mid_point(opp)]))[0]
        w_o = ox if vertical else oy
        if pair:
            m1, b1 = _fit_in(H, p1)
            m2, b2 = _fit_in(H, p2)

            def sep_at(w):
                return abs((m2 * w + b2) - (m1 * w + b1))
            guess = w_o + sign * along_mm * sep_at(w_o) / perp_mm
            ppm_f = sep_at(0.5 * (w_o + guess)) / perp_mm
        else:
            ppm_f = frame_ppm
        if not ppm_f > 0:
            return None, "the edges either side cross over"
        offset = sign * along_mm * ppm_f
        ends_fr = G.transform_points(H, opp.points(len(us)))
        uf = ends_fr[:, 1] if vertical else ends_fr[:, 0]
        vf = mo * uf + bo + offset
        pts_fr = np.stack([vf, uf] if vertical else [uf, vf], axis=1)
        pts_ph = G.transform_points(Hinv, pts_fr)
        if not np.all(np.isfinite(pts_ph)):
            return None, "it could not be placed back on the photo"
        uu, vv = ((pts_ph[:, 1], pts_ph[:, 0]) if vertical
                  else (pts_ph[:, 0], pts_ph[:, 1]))
        v_new = np.polyval(np.polyfit(uu, vv, 1), us)
        # local photo scale at the new edge: map 1mm across it back
        wn = float(np.median(vf))
        un = float(np.median(uf))
        a = np.array([[wn, un], [wn + ppm_f, un]] if vertical
                     else [[un, wn], [un, wn + ppm_f]])
        a_ph = G.transform_points(Hinv, a)
        ppm_here = float(np.linalg.norm(a_ph[1] - a_ph[0]))
    else:
        def perp_sep(w):
            return float(p2.v_at(w) - p1.v_at(w))
        v_opp = opp.v_at(us)
        v_new = np.empty_like(v_opp)
        for i, vo in enumerate(v_opp):
            v = vo + sign * along_mm * perp_sep(vo) / perp_mm
            for _ in range(8):
                # linear scale: mean scale over [vo, v] = scale at midpoint
                v_next = vo + sign * along_mm * \
                    perp_sep(0.5 * (vo + v)) / perp_mm
                if abs(v_next - v) < 1e-4:
                    v = v_next
                    break
                v = v_next
            v_new[i] = v
        v_mid = float(np.median(v_new))
        ppm_here = perp_sep(v_mid) / perp_mm
        if not ppm_here > 0:
            return None, "the edges either side cross over"
        # keystone of the pair -> tilt about the missing edge's direction
        f_px = ASSUMED_FOCAL_MM_EQUIV / 43.266 * math.hypot(W, Himg)
        v_opp_mid = float(np.median(v_opp))
        span_px = abs(v_mid - v_opp_mid)
        s_a, s_b = perp_sep(v_opp_mid), perp_sep(v_mid)
        grad = abs(s_b - s_a) / (0.5 * (s_a + s_b)) / max(span_px, 1.0)
        tilt = max(math.degrees(math.atan(f_px * grad)), MIN_TILT_DEG)

    if not np.all(np.isfinite(v_new)):
        return None, "the construction did not converge"
    # the constructed edge must lie inside the photo (with room for the
    # scanner's profile); otherwise an input edge is badly wrong
    limit = W if vertical else Himg
    if v_new.min() < 2 or v_new.max() > limit - 3:
        return None, ("working it out from the other edges puts it "
                      "outside the photo, so one of those edges must be "
                      "wrong")
    line = G.FittedLine("v" if vertical else "h",
                        *np.polyfit(us, v_new, 1),
                        rms=opp.rms, n=opp.n, n_rej=0,
                        u_range=(float(lo), float(hi)), bow_px=None)

    # --- uncertainty terms (mm, 1 sigma) ---
    if pair:
        # the constructed distance and the perpendicular size that
        # converts px to mm are both manufactured dimensions
        t_size = along_mm * math.hypot(CUT_SIZE_TOL_MM / along_mm,
                                       CUT_SIZE_TOL_MM / perp_mm)
        t_scale = along_mm * sep_unc_mm / perp_mm
    else:
        t_size = CUT_SIZE_TOL_MM
        t_scale = along_mm * frame_ppm_rel_unc
    if H is not None:
        t_geo, geo_name = h_unc_mm, "alignment"
    else:
        t_geo = along_mm * (1.0 - math.cos(math.radians(tilt)))
        geo_name = "tilt"
    extra = math.sqrt(t_size ** 2 + t_scale ** 2 + t_geo ** 2
                      + SHAPE_UNC_MM ** 2 + opp_unc_mm ** 2)
    terms = {"size": t_size, "scale": t_scale, geo_name: t_geo,
             "shape": SHAPE_UNC_MM, "opposite_edge": opp_unc_mm}
    dim = "width" if vertical else "height"
    how = ("" if pair else ", scaled by the official picture")
    note = (f"worked out from the {OPPOSITE[side]} edge and the card's "
            f"{along_mm:.1f}mm {dim}{how}, good to about {extra:.2f}mm")
    return InferredEdge(line, extra, note, tilt, ppm_here, terms,
                        "pair" if pair else "picture"), ""
