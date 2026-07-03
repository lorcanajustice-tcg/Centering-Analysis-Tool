"""Error-budget composition helpers."""
from __future__ import annotations

import math

from .types import Uncertainty


def ratio_pts_sigma(a_mm: float, b_mm: float,
                    sa_mm: float, sb_mm: float) -> float:
    """Sigma (in ratio points) of pct = 100*a/(a+b) given independent border
    sigmas."""
    s = a_mm + b_mm
    return 100.0 * math.sqrt((b_mm * sa_mm) ** 2 + (a_mm * sb_mm) ** 2) / s**2


def border_stat_sigma_px(edge_rms: float, edge_n: int,
                         frame_rms: float, frame_n: int) -> float:
    """Statistical sigma of a border width (two independent robust line
    fits)."""
    return math.sqrt(edge_rms**2 / max(edge_n, 1) +
                     frame_rms**2 / max(frame_n, 1))


def compose_ratio_uncertainty(a_mm, b_mm, stat_a_mm, stat_b_mm,
                              def_a_mm, def_b_mm, perspective_pts) -> Uncertainty:
    return Uncertainty(
        statistical=ratio_pts_sigma(a_mm, b_mm, stat_a_mm, stat_b_mm),
        perspective=perspective_pts,
        edge_definition=ratio_pts_sigma(a_mm, b_mm, def_a_mm, def_b_mm),
    )
