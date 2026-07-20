"""Estimate-tier edge rescue.

When the strict tier refuses an edge (too few scan lines, or detections
too inconsistent to fit), this module attempts an EXPLICITLY LABELLED
estimate instead of silently giving up - without eroding the project
never-guess ethos:

- estimates are always status="estimated", never "measured";
- every rescue carries an inflated uncertainty term (extra_unc_mm);
- the axis measurement is refused outright when its total uncertainty
  exceeds CAP_MM (a number wider than ~a grading band is noise);
- ambiguity (multiple well-separated candidate lines that the hybrid
  cut cross-check cannot arbitrate) stays refused.

Mechanics: the accepted scan-line detections are split into clusters by
position - a refused edge often hides one clean edge plus a parallel
contaminant (sleeve edge, glare boundary, shadow line). Each internally
consistent cluster is fitted; the hybrid cut_scan detector
(fitting._hybrid_cross_check), which keys on the card-interior plateau and
is immune to the shadow-band artifact, arbitrates which cluster is the
physical cut.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import geometry as G
from .fitting import _hybrid_cross_check

# refusal floor: estimated axis results wider than this are refused
CAP_MM = 0.5
# the hybrid cut cross-check must agree with a candidate within this
HYBRID_AGREE_MM = 0.30
MIN_POINTS = 4
FIT_RMS_MAX_PX = 1.5
MIN_SPAN_FRAC = 0.30
# floor systematic for any estimate-tier edge (unmodelled contamination)
BASE_SYSTEMATIC_MM = 0.10


@dataclass
class EdgeEstimate:
    line: G.FittedLine
    extra_unc_mm: float
    note: str


def _clusters(u: np.ndarray, v: np.ndarray, gap_px: float):
    order = np.argsort(v)
    us, vs = u[order], v[order]
    cuts = np.where(np.diff(vs) > gap_px)[0]
    return list(zip(np.split(us, cuts + 1), np.split(vs, cuts + 1)))


def rescue_edge(gray, side, u_det, v_det, us_grid, ppm):
    """Attempt an estimate-tier rescue of a strict-refused edge.

    u_det/v_det: RAW accepted per-line detections from the strict scan
    (before line fitting). us_grid: the full scan grid (for span/anchor).

    Returns (EdgeEstimate, "") on success or (None, reason) on refusal.
    """
    u_det = np.asarray(u_det, dtype=float)
    v_det = np.asarray(v_det, dtype=float)
    us_grid = np.asarray(us_grid, dtype=float)
    if len(u_det) < MIN_POINTS:
        return None, f"only {len(u_det)} raw detections (need >= {MIN_POINTS})"
    span_ref = float(us_grid.max() - us_grid.min())
    if span_ref <= 0 or not ppm:
        return None, "degenerate scan grid"
    gap_px = max(4.0, 0.15 * ppm)
    orientation = "v" if side in ("left", "right") else "h"
    cands = []
    for cu, cv in _clusters(u_det, v_det, gap_px):
        if len(cu) < MIN_POINTS:
            continue
        span = float(cu.max() - cu.min()) / span_ref
        if span < MIN_SPAN_FRAC:
            continue
        line = G.FittedLine.fit(orientation, cu, cv)
        if line.rms is None or line.rms > FIT_RMS_MAX_PX:
            continue
        cands.append((line, len(cu), span))
    if not cands:
        return None, (f"no internally consistent cluster among {len(u_det)} "
                      f"detections (need >= {MIN_POINTS} points, span >= "
                      f"{MIN_SPAN_FRAC:.0%}, rms <= {FIT_RMS_MAX_PX}px)")
    survivors = []
    for line, ncl, span in cands:
        d = _hybrid_cross_check(gray, side, line, us_grid, ppm)
        if d is not None and abs(d) <= HYBRID_AGREE_MM:
            survivors.append((line, ncl, span, d))
    if not survivors:
        return None, (f"hybrid cut cross-check confirms none of {len(cands)} "
                      "candidate cluster(s)")
    if len(survivors) > 1:
        mid = float(np.median(us_grid))
        vs_mid = [float(s[0].v_at(mid)) for s in survivors]
        if (max(vs_mid) - min(vs_mid)) / ppm > HYBRID_AGREE_MM:
            return None, (f"{len(survivors)} well-separated clusters each "
                          "pass the hybrid cross-check; ambiguous")
        survivors.sort(key=lambda s: -s[1])
    line, ncl, span, d = survivors[0]
    lever = (line.rms / max(math.sqrt(line.n), 1.0)) / ppm / max(span, 1e-6)
    extra = math.sqrt(BASE_SYSTEMATIC_MM ** 2 + d ** 2 + lever ** 2)
    note = (f"estimated from a {ncl}-point cluster of {len(u_det)} raw "
            f"detections (span {span:.0%}, rms {line.rms:.2f}px); hybrid "
            f"cut cross-check agrees within {abs(d):.2f}mm; extra "
            f"systematic +-{extra:.2f}mm")
    return EdgeEstimate(line, extra, note), ""
