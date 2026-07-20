"""Shared edge-fit machinery: per-edge fit reports + refusal gates, and
the hybrid cut cross-check (shadow-band detection).

Moved verbatim from back.py (2026-07-20) so stage 5 (fitting & gates) is
a standalone component used by BOTH pipelines; back.py re-exports the
names for backward compatibility. Zero behaviour change.
"""
from __future__ import annotations

import numpy as np

from . import edges as E
from . import geometry as G
from .types import EdgeFitReport, QAFlag


# hybrid cross-check: displacement beyond which a shadow band is suspected
# (the documented artifact displaces step/texture scans by ~0.5-0.7mm;
# threshold sits well above the +-0.1mm accuracy target)
SHADOW_BAND_MM = 0.25


def _hybrid_cross_check(gray, side, line, us, ppm):
    """Signed displacement (mm) of the hybrid cut detector vs the fitted
    edge, positive INWARD (hybrid places the cut inside the primary
    detection - the shadow-band signature). None when unavailable.

    The window reaches 1.5mm inside the primary edge with the plateau
    sampled at 1.5..1.0mm inside, so a cut dragged outward by up to ~1mm
    is still caught with an uncontaminated interior plateau."""
    try:
        hu, hv, hd = E.cut_scan(gray, side, line.v_at, us, ppm,
                                win_out_mm=1.5, win_in_mm=1.5,
                                plateau_mm=-1.0)
    except ValueError:
        return None
    if hd.n_ok < 15:
        return None
    orientation = "v" if side in ("left", "right") else "h"
    hline = G.FittedLine.fit(orientation, hu, hv)
    if hline.rms > 3.0:
        return None
    mid = float(np.median(us))
    disp = (float(hline.v_at(mid)) - float(line.v_at(mid))) / ppm
    inward = 1.0 if side in ("left", "top") else -1.0
    return disp * inward


def _shadow_band_qa(qa, gray, lines, rows, cols, ppm):
    """Run the hybrid cross-check on every fitted edge; QA-flag sides whose
    hybrid cut sits > SHADOW_BAND_MM from the primary scan. Cross-check
    only: measurements are NOT altered (calibration reshoot pending)."""
    for side, line in lines.items():
        if line is None:
            continue
        us = rows if side in ("left", "right") else cols
        d = _hybrid_cross_check(gray, side, line, us, ppm)
        if d is not None and abs(d) > SHADOW_BAND_MM:
            where = "inside" if d > 0 else "outside"
            qa.append(QAFlag(
                "SHADOW_BAND_SUSPECTED",
                f"{side} edge: hybrid cut detector places the cut "
                f"{abs(d):.2f}mm {where} the primary detection; a shadow "
                "band or glare may be displacing the edge scan - distrust "
                "this side and prefer a diffuse-light recapture",
                severity="warning"))


def _edge_report(name, method, us, vs, diag, flag_rms=1.5, min_pts=10):
    rep = EdgeFitReport(edge=name, method=method, n_points=diag.n_ok,
                        n_rejected=diag.n_attempted - diag.n_ok)
    if diag.n_ok < min_pts:
        rep.status = "refused"
        rep.notes.append(f"only {diag.n_ok}/{diag.n_attempted} scan lines "
                         f"usable ({diag.summary()})")
        return None, rep
    orientation = "v" if name.endswith(("left", "right")) else "h"
    line = G.FittedLine.fit(orientation, us, vs)
    rep.n_points = line.n
    rep.n_rejected += line.n_rej
    rep.rms_residual_px = line.rms
    rep.angle_deg = line.angle_from_nominal_deg()
    rep.bow_px = line.bow_px
    if line.rms > 4.0:
        rep.status = "refused"
        rep.notes.append(f"fit residual {line.rms:.2f}px far above target; "
                         "detections inconsistent")
        return None, rep
    if line.rms > flag_rms:
        rep.status = "flagged"
        rep.notes.append(f"fit residual {line.rms:.2f}px above {flag_rms}px target")
    if diag.n_ok < 0.7 * diag.n_attempted:
        rep.notes.append(f"partial coverage: {diag.summary()}")
    return line, rep
