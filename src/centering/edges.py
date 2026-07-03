"""Edge scanners.

Three detectors, per the proven prototype methodology:
- texture_scan: dark card on dark mat. Brightness fails; mat grain gives high
  local std-dev, smooth card border low. Edge = sustained drop below the
  midpoint of outer-median vs inner-8th-percentile texture.
- step_scan: bright card on dark mat. Sub-pixel 50%-threshold crossing with
  linear interpolation; low-contrast lines are excluded, not tolerated.
- frame_peak_scan: printed bright frame line; first peak scanning inward from
  the detected card edge (avoids inner decorative doubled lines); sub-pixel
  centre by intensity-weighted centroid.

All scanners work on near-axis-aligned edges (valid for tilt < ~5 deg) and
return per-line sub-pixel positions in image coordinates plus diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ScanDiagnostics:
    n_attempted: int = 0
    n_ok: int = 0
    reject_reasons: dict = field(default_factory=dict)
    median_transition_px: float | None = None

    n_snapped: int = 0

    def note_reject(self, reason: str):
        self.reject_reasons[reason] = self.reject_reasons.get(reason, 0) + 1

    def note_snap(self):
        self.n_snapped += 1

    def summary(self) -> str:
        parts = [f"{v}x {k}" for k, v in sorted(self.reject_reasons.items())]
        return ", ".join(parts) if parts else "none"


def _profile_band(gray: np.ndarray, side: str, u: float, lo: float, hi: float,
                  band: int):
    """1-D profile stack across a band perpendicular to the edge.

    Returns (band_2d [band x L], coords [L]) where coords are the image
    coordinate along the scan direction, ALWAYS ordered mat(outside)->card.
    """
    H, W = gray.shape
    u = int(round(u))
    hb = band // 2
    lo_i, hi_i = int(max(0, lo)), int(min((W if side in ("left", "right") else H), hi))
    if side in ("left", "right"):
        r0, r1 = max(0, u - hb), min(H, u + hb)
        block = gray[r0:r1, lo_i:hi_i]
        coords = np.arange(lo_i, hi_i, dtype=np.float64)
    else:
        c0, c1 = max(0, u - hb), min(W, u + hb)
        block = gray[lo_i:hi_i, c0:c1].T
        coords = np.arange(lo_i, hi_i, dtype=np.float64)
    if side in ("right", "bottom"):
        block = block[:, ::-1]
        coords = coords[::-1]
    return block, coords


def _smooth(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    k = np.ones(w) / w
    return np.convolve(x, k, mode="same")


def texture_scan(gray: np.ndarray, side: str, approx: float,
                 scan_us: np.ndarray, search_out_px: float, search_in_px: float,
                 band: int = 24, sustain: int = 18, min_sep: float = 6.0,
                 smooth_w: int = 5):
    """Texture-transition edge detection (dark on dark).

    Returns (us, vs, diag): per accepted scan line the coordinate along the
    edge (us) and sub-pixel edge position (vs).
    """
    us, vs = [], []
    diag = ScanDiagnostics()
    trans = []
    for u in scan_us:
        diag.n_attempted += 1
        if side in ("left", "top"):
            lo, hi = approx - search_out_px, approx + search_in_px
        else:
            lo, hi = approx - search_in_px, approx + search_out_px
        block, coords = _profile_band(gray, side, u, lo, hi, band)
        if block.shape[0] < band // 2 or block.shape[1] < 60:
            diag.note_reject("band_truncated")
            continue
        prof = _smooth(block.std(axis=0), smooth_w)
        n = len(prof)
        outer = float(np.median(prof[: max(10, n // 5)]))
        inner = float(np.percentile(prof[-max(10, n // 5):], 8))
        if outer - inner < min_sep:
            diag.note_reject("insufficient_texture_contrast")
            continue
        thr = 0.5 * (outer + inner)
        below = prof < thr
        # first index where 'below' holds for `sustain` consecutive samples
        idx = None
        run = 0
        for i, b in enumerate(below):
            run = run + 1 if b else 0
            if run >= sustain:
                idx = i - sustain + 1
                break
        if idx is None or idx == 0:
            diag.note_reject("no_sustained_crossing")
            continue
        # sub-pixel: linear interpolation across the threshold
        p0, p1 = prof[idx - 1], prof[idx]
        frac = 0.0 if p0 == p1 else float(np.clip((p0 - thr) / (p0 - p1), 0, 1))
        pos = coords[idx - 1] + frac * (coords[idx] - coords[idx - 1])
        # transition width (80%..20% of the outer-inner drop)
        hi_lvl = inner + 0.8 * (outer - inner)
        lo_lvl = inner + 0.2 * (outer - inner)
        j0 = idx - 1
        while j0 > 0 and prof[j0] < hi_lvl:
            j0 -= 1
        j1 = idx
        while j1 < n - 1 and prof[j1] > lo_lvl:
            j1 += 1
        trans.append(abs(j1 - j0))
        us.append(float(u))
        vs.append(float(pos))
        diag.n_ok += 1
    if trans:
        diag.median_transition_px = float(np.median(trans))
    return np.array(us), np.array(vs), diag


def step_scan(gray: np.ndarray, side: str, approx: float, scan_us: np.ndarray,
              search_out_px: float, search_in_px: float, band: int = 3,
              min_contrast: float = 30.0, sustain: int = 6):
    """Sub-pixel 50%-threshold brightness step (bright card on dark mat)."""
    us, vs = [], []
    diag = ScanDiagnostics()
    trans = []
    for u in scan_us:
        diag.n_attempted += 1
        if side in ("left", "top"):
            lo, hi = approx - search_out_px, approx + search_in_px
        else:
            lo, hi = approx - search_in_px, approx + search_out_px
        block, coords = _profile_band(gray, side, u, lo, hi, band)
        if block.shape[1] < 40:
            diag.note_reject("band_truncated")
            continue
        prof = block.mean(axis=0)
        n = len(prof)
        outer = float(np.median(prof[: max(8, n // 4)]))
        inner = float(np.median(prof[-max(8, n // 4):]))
        if inner - outer < min_contrast:
            diag.note_reject("insufficient_contrast")
            continue
        thr = 0.5 * (outer + inner)
        above = prof > thr
        # First sustained crossing whose inner side actually STAYS at card
        # level: a glare band on the mat crosses the threshold but dips back
        # down before the true edge, so re-search past failed candidates.
        idx = None
        run = 0
        i = 0
        while i < n:
            run = run + 1 if above[i] else 0
            if run >= sustain:
                cand = i - sustain + 1
                tail = prof[cand:min(cand + 30, n)]
                if np.median(tail) >= inner - 0.3 * (inner - outer):
                    idx = cand
                    break
                diag.note_reject("glare_band_skipped")
                run = 0
            i += 1
        if idx is None or idx == 0:
            diag.note_reject("no_crossing")
            continue
        p0, p1 = prof[idx - 1], prof[idx]
        if p1 <= p0:
            diag.note_reject("non_monotonic_at_edge")
            continue
        frac = float(np.clip((thr - p0) / (p1 - p0), 0, 1))
        pos = coords[idx - 1] + frac * (coords[idx] - coords[idx - 1])
        lo_lvl = outer + 0.2 * (inner - outer)
        hi_lvl = outer + 0.8 * (inner - outer)
        j0 = idx - 1
        while j0 > 0 and prof[j0] > lo_lvl:
            j0 -= 1
        j1 = idx
        while j1 < n - 1 and prof[j1] < hi_lvl:
            j1 += 1
        trans.append(abs(j1 - j0))
        us.append(float(u))
        vs.append(float(pos))
        diag.n_ok += 1
    if trans:
        diag.median_transition_px = float(np.median(trans))
    return np.array(us), np.array(vs), diag


def frame_peak_scan(gray: np.ndarray, side: str, edge_line, scan_us: np.ndarray,
                    px_per_mm: float, min_peak: float = 45.0,
                    search_mm: tuple = (0.5, 6.0), centroid_half: int = 5,
                    band: int = 3):
    """First bright peak scanning INWARD from the detected card edge.

    edge_line: FittedLine of the physical card edge for this side; scanning
    starts from its per-row position, which anchors past the sleeve/edge blur
    and stops decorative inner doubled lines from being picked up.
    """
    us, vs = [], []
    diag = ScanDiagnostics()
    for u in scan_us:
        diag.n_attempted += 1
        e = float(edge_line.v_at(u))
        s0, s1 = search_mm[0] * px_per_mm, search_mm[1] * px_per_mm
        if side in ("left", "top"):
            lo, hi = e + s0, e + s1
        else:
            lo, hi = e - s1, e - s0
        block, coords = _profile_band(gray, side, u, lo, hi, band)
        if block.shape[1] < 20:
            diag.note_reject("band_truncated")
            continue
        prof = block.mean(axis=0)
        base = float(np.median(prof))
        exc = prof - base
        cand = np.where(exc >= min_peak)[0]
        if len(cand) == 0:
            diag.note_reject("no_peak_above_threshold")
            continue
        # first contiguous group scanning inward (profile is ordered out->in
        # only for left/top; for right/bottom _profile_band already flipped
        # it so index 0 is the OUTER end -- which is what we want: first
        # peak encountered moving inward from the edge)
        first = cand[0]
        grp_end = first
        while grp_end + 1 < len(prof) and exc[grp_end + 1] >= min_peak * 0.5:
            grp_end += 1
        pk = first + int(np.argmax(exc[first:grp_end + 1]))
        a, b = max(0, pk - centroid_half), min(len(prof), pk + centroid_half + 1)
        w = np.clip(exc[a:b], 0, None)
        if w.sum() <= 0:
            diag.note_reject("degenerate_peak")
            continue
        pos = float(np.sum(coords[a:b] * w) / w.sum())
        us.append(float(u))
        vs.append(pos)
        diag.n_ok += 1
    return np.array(us), np.array(vs), diag
