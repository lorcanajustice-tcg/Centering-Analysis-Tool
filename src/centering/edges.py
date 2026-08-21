"""Edge scanners.

Three detectors, per the proven prototype methodology:
- texture_scan: dark card on dark mat. Brightness fails; mat grain gives high
  local std-dev, smooth card border low. Edge = sustained drop below the
  midpoint of outer-median vs inner-8th-percentile texture.
- step_scan: brightness step in EITHER polarity (bright card on dark mat, or
  dark card / dark border on a light background). Sub-pixel 50%-threshold
  crossing with linear interpolation; low-contrast lines are excluded, not
  tolerated.
- colour_scan: chromaticity step. On a COLOURED surface the shadow a card
  casts is a brightness ramp that starts outside the cut, so a brightness
  threshold lands outside it; hue survives the shadow, so the hue step is
  the cut. Refuses when background and card share a hue (white paper), and
  the caller falls back.
- frame_peak_scan: printed bright frame line; first peak scanning inward from
  the detected card edge (avoids inner decorative doubled lines); sub-pixel
  centre by intensity-weighted centroid.
- cut_scan: hybrid specular-ridge / plateau-knee cut detector, ported from the
  calibration prototype (calibration/measure_crop_bias.py). Keys on the
  departure from the card-interior plateau rather than a mid-level threshold,
  so shadow bands hugging the card edge do not drag it outward; used as a
  cross-check against step/texture scans.

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
    method_counts: dict = field(default_factory=dict)

    def note_method(self, method: str):
        self.method_counts[method] = self.method_counts.get(method, 0) + 1

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


def _outside_count(coords: np.ndarray, side: str, approx: float) -> int:
    """Samples of a scan profile that lie OUTSIDE the nominal card edge.

    `_profile_band` clips its window to the image, so the outside span can
    be far shorter than `search_out_px` asked for - on a scan or a tightly
    cropped photo it may be a few dozen pixels. The background/card level
    estimators below size their sampling windows from this so they never
    reach across the edge (which would flatten the contrast and refuse a
    perfectly crisp edge).
    """
    outside = coords < approx if side in ("left", "top") else coords > approx
    return int(np.count_nonzero(outside))


def _level_windows(n: int, n_out: int, frac_div: int, floor: int = 8):
    """(w_out, w_in) sample counts for the outer/inner level estimates.

    Historical behaviour was a fixed n//frac_div at each end; that is kept
    whenever the spans are long enough to support it, so results on
    normally-framed photos are bit-identical.
    """
    w = max(floor, n // frac_div)
    w_out = max(floor, min(w, int(0.8 * n_out)))
    w_in = max(floor, min(w, int(0.8 * (n - n_out))))
    return w_out, w_in


def texture_scan(gray: np.ndarray, side: str, approx: float,
                 scan_us: np.ndarray, search_out_px: float, search_in_px: float,
                 band: int = 24, sustain: int = 18, min_sep: float = 6.0,
                 smooth_w: int = 5, allow_inverted: bool = True):
    """Texture-transition edge detection (dark on dark).

    Returns (us, vs, diag): per accepted scan line the coordinate along the
    edge (us) and sub-pixel edge position (vs).

    `allow_inverted` enables the smooth-surround / textured-face polarity
    (see below). It belongs to the FINE pass, which searches a window
    around a known edge: in a wide-open coarse search the inverted
    polarity will happily report the first texture change inside the
    artwork, so the coarse pass keeps the historical polarity only.
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
        w_out, w_in = _level_windows(n, _outside_count(coords, side, approx),
                                     5, floor=10)
        # polarity-agnostic, mirroring step_scan: the historical case is a
        # textured mat against a smooth card border, but a smooth surround
        # (scanner platen, out-of-focus paper, the flat colour a crop is
        # padded with) against a grainy printed face is the same signal
        # inverted. The smooth side is summarised by its 8th percentile
        # (robust to specks), the textured side by its median; the profile
        # is then normalised so the smooth side is always HIGH and the
        # historical "sustained drop" logic applies unchanged. For the
        # historical polarity pol=+1 and the arithmetic is unchanged.
        # the flip is only taken when the inverted contrast is unambiguous
        # (inside rougher than outside by at least the separation the
        # detector would demand anyway); marginal lines keep the
        # historical polarity, so dark-mat behaviour is untouched
        m_out = float(np.median(prof[:w_out]))
        m_in = float(np.median(prof[-w_in:]))
        pol = -1.0 if (allow_inverted and m_in - m_out > min_sep) else 1.0
        if pol > 0:
            outer = float(np.median(prof[:w_out]))
            inner = float(np.percentile(prof[-w_in:], 8))
        else:
            outer = -float(np.percentile(prof[:w_out], 8))
            inner = -float(np.median(prof[-w_in:]))
            prof = -prof
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
    """Sub-pixel 50%-threshold brightness step, polarity-agnostic.

    Handles bright card on dark mat AND dark card (or dark border) on a
    light background: the profile is normalized so the card side is always
    the HIGH level, then the proven bright-on-dark logic applies unchanged.
    For bright-on-dark input the arithmetic is bit-identical (polarity +1).
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
        if block.shape[1] < 40:
            diag.note_reject("band_truncated")
            continue
        prof = block.mean(axis=0)
        n = len(prof)
        w_out, w_in = _level_windows(n, _outside_count(coords, side, approx), 4)
        outer = float(np.median(prof[:w_out]))
        inner = float(np.median(prof[-w_in:]))
        if abs(inner - outer) < min_contrast:
            diag.note_reject("insufficient_contrast")
            continue
        # normalize so the card (inner) side is the high level; for the
        # historical bright-on-dark case pol=+1 and nothing changes
        pol = 1.0 if inner >= outer else -1.0
        prof = pol * prof
        outer, inner = pol * outer, pol * inner
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
        # Shadow-penumbra guard: the plateau just OUTSIDE an accepted
        # crossing must sit at the background level. A soft shadow (or
        # glare penumbra) between background and card shifts it toward
        # the card level, and the 50% threshold then lands inside the
        # penumbra - up to ~1.5mm outside the cut. Such lines are
        # excluded honestly (the fit uses the clean span, or the side is
        # refused when too few lines remain). Lines whose whole outside
        # IS the shadow still measure: their `outer` level is the shadow
        # and the crossing is the true cut. Accepted lines are
        # bit-identical to the historical behaviour.
        if idx >= 12:
            near_out = float(np.median(prof[idx - 10:idx - 2]))
            if abs(near_out - outer) > 0.35 * (inner - outer):
                diag.note_reject("shadowed_outside_level")
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


def chromaticity(rgb: np.ndarray) -> np.ndarray:
    """Illumination-invariant 2-plane chromaticity, float32 (H, W, 2).

    r/(r+g+b), g/(r+g+b). Scaling all three channels by the same factor -
    what a shadow, a vignette or an exposure change does to first order -
    leaves both planes unchanged, so the background's chromaticity is the
    same in the shadow hugging a card edge as it is out in the open.
    Brightness is not: that is the whole point.
    """
    f = rgb.astype(np.float32)
    s = f.sum(axis=2) + 1e-3
    return np.stack([f[:, :, 0] / s, f[:, :, 1] / s], axis=2)


def colour_scan(chroma: np.ndarray, side: str, approx: float,
                scan_us: np.ndarray, search_out_px: float, search_in_px: float,
                band: int = 3, min_sep: float = 0.12, snr_min: float = 8.0,
                sustain: int = 6):
    """Sub-pixel card edge from the CHROMATICITY step, not the brightness step.

    Motivation (2026-08-21, TAG scans): a card lying on a coloured surface
    casts a soft shadow onto that surface. In brightness that shadow is a
    ramp tens of pixels wide starting outside the cut, and a 50%-of-
    (background, card) threshold lands part-way up it - OUTSIDE the cut.
    Measured on two TAG scans: the brightness edge sits 4.7-5.3px (~0.07mm)
    outside the chromatic edge on every side of a card back, and 8.8-9.6px
    outside on the LEFT of a card front while sitting ~0.8px INSIDE on the
    right (that face shows a dark cut-edge rim on one side and a bright one
    on the other). The symmetric part inflates both margins; the asymmetric
    part moves the card centre, and with it the centering ratio.

    Shadowed background keeps its hue, so the chromatic profile stays flat
    right up to the cut. `chroma` is the output of `chromaticity()`.
    Needs a background whose hue differs from the card's: a refusal means
    "no colour signal here", and the caller falls back to brightness.

    Gates. `min_sep` is a distance in the normalised (r, g) plane, where
    the achromatic point sits at (1/3, 1/3) and a fully saturated primary
    ~0.5 away; 0.12 therefore asks the two surfaces to differ by about a
    quarter of the maximum possible chromatic contrast - a different hue,
    not a slightly tinted grey. Measured: the TAG scans' orange backing
    against this card reaches 0.18-0.31 on every side of all four scans,
    while the white-paper fixture captures reach 0.001-0.10, so the gate
    has ~2x headroom on both sides of a real gap. `snr_min` additionally
    requires the step to stand clear of the chroma noise on the background
    side, which is what protects a dark, noisy capture on a coloured mat.
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
        b0, coords = _profile_band(chroma[:, :, 0], side, u, lo, hi, band)
        b1, _ = _profile_band(chroma[:, :, 1], side, u, lo, hi, band)
        if b0.shape[1] < 40:
            diag.note_reject("band_truncated")
            continue
        p0, p1 = b0.mean(axis=0), b1.mean(axis=0)
        n = len(p0)
        w_out, w_in = _level_windows(n, _outside_count(coords, side, approx), 4)
        o0, o1 = float(np.median(p0[:w_out])), float(np.median(p1[:w_out]))
        i0, i1 = float(np.median(p0[-w_in:])), float(np.median(p1[-w_in:]))
        d0, d1 = i0 - o0, i1 - o1
        sep = float(np.hypot(d0, d1))
        if sep < min_sep:
            diag.note_reject("insufficient_chroma")
            continue
        # project onto the background->card chromaticity axis: 0 outside,
        # `sep` inside, monotone across the cut whatever the hues are
        prof = (p0 - o0) * (d0 / sep) + (p1 - o1) * (d1 / sep)
        noise = 1.4826 * float(np.median(np.abs(prof[:w_out]
                                                - np.median(prof[:w_out]))))
        if sep < snr_min * noise:
            diag.note_reject("chroma_step_below_noise")
            continue
        thr = 0.5 * sep
        above = prof > thr
        idx, run, i = None, 0, 0
        while i < n:
            run = run + 1 if above[i] else 0
            if run >= sustain:
                cand = i - sustain + 1
                tail = prof[cand:min(cand + 30, n)]
                if np.median(tail) >= 0.7 * sep:
                    idx = cand
                    break
                diag.note_reject("no_card_plateau")
                run = 0
            i += 1
        if idx is None or idx == 0:
            diag.note_reject("no_crossing")
            continue
        a, b = prof[idx - 1], prof[idx]
        if b <= a:
            diag.note_reject("non_monotonic_at_edge")
            continue
        frac = float(np.clip((thr - a) / (b - a), 0, 1))
        pos = coords[idx - 1] + frac * (coords[idx] - coords[idx - 1])
        j0 = idx - 1
        while j0 > 0 and prof[j0] > 0.2 * sep:
            j0 -= 1
        j1 = idx
        while j1 < n - 1 and prof[j1] < 0.8 * sep:
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
                    band: int = 3, rescue_floor: float = 0.45,
                    modal_window_mm: float = 0.25):
    """First bright peak scanning INWARD from the detected card edge.

    edge_line: FittedLine of the physical card edge for this side; scanning
    starts from its per-row position, which anchors past the sleeve/edge blur
    and stops decorative inner doubled lines from being picked up.

    modal_window_mm: detections further than this from the MODAL cut-to-line
    distance are dropped before the fit - they are decorative structure
    crossing the search band, not the frame line.

    rescue_floor: when NO peak clears the absolute `min_peak`, the scan line
    is retried against a prominence threshold (half the window's strongest
    excess, floored at rescue_floor*min_peak) rather than refused. Lines that
    already found a peak never reach it.
    """
    us, vs, offs = [], [], []
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
        thr = min_peak
        cand = np.where(exc >= thr)[0]
        if len(cand) == 0:
            # Rescue pass. min_peak is an ABSOLUTE brightness excess, so a
            # frame line printed or scanned a little duller than the
            # calibration card's refuses the whole side even when it is
            # plainly the dominant feature in the window: the 2026-08-21 TAG
            # back's top line reaches 43 against a min_peak of 45 and its
            # T/B was refused outright. Retry keyed on PROMINENCE instead -
            # the peak must still stand clear of everything else in the
            # window. Strictly additive: a side that already found a peak
            # never reaches this branch, so its result is unchanged.
            rescue = max(0.5 * float(exc.max()), min_peak * rescue_floor)
            cand = np.where(exc >= rescue)[0]
            if len(cand) == 0:
                diag.note_reject("no_peak_above_threshold")
                continue
            thr = rescue
            diag.note_reject("min_peak_rescued")
        # first contiguous group scanning inward (profile is ordered out->in
        # only for left/top; for right/bottom _profile_band already flipped
        # it so index 0 is the OUTER end -- which is what we want: first
        # peak encountered moving inward from the edge)
        first = cand[0]
        grp_end = first
        while grp_end + 1 < len(prof) and exc[grp_end + 1] >= thr * 0.5:
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
        offs.append(abs(pos - e))
        diag.n_ok += 1
    us, vs, offs = np.array(us), np.array(vs), np.array(offs)
    if len(us) >= 8 and modal_window_mm:
        # A printed frame line runs PARALLEL to the cut: its distance from
        # the card edge is the same all along the side (a print skewed
        # against the cut varies it by ~0.1mm across a card, far inside the
        # window below). Decorative structure crossing the search band - the
        # Lorcana back's corner chevrons and its centre notch, which between
        # them own about a third of the top and bottom scan lines - is a
        # different feature at a different distance, and enough of it drags
        # the LTS fit into an "inconsistent detections" refusal. Keep the
        # modal distance and drop the rest, so the line fit sees the one
        # feature it assumes it is looking at.
        binw = 0.05 * px_per_mm
        bins = np.arange(offs.min(), offs.max() + binw, binw)
        if len(bins) >= 2:
            hist, _ = np.histogram(offs, bins=bins)
            k = int(np.argmax(hist))
            mode = 0.5 * (bins[k] + bins[k + 1])
            keep = np.abs(offs - mode) <= modal_window_mm * px_per_mm
            if keep.sum() >= 8:
                diag.n_ok = int(keep.sum())
                for _ in range(int((~keep).sum())):
                    diag.note_reject("off_modal_frame_distance")
                us, vs = us[keep], vs[keep]
    return us, vs, diag


def _dir_profile(gray32: np.ndarray, p, n_dir, offs: np.ndarray, band: int,
                 e_dir) -> np.ndarray:
    """Band-averaged profile at point p sampled along n_dir at offsets offs
    (sub-pixel bilinear); `band` parallel samples averaged along e_dir."""
    import cv2
    k = np.arange(-(band // 2), band // 2 + 1, dtype=np.float64)
    xs = p[0] + offs[:, None] * n_dir[0] + k[None, :] * e_dir[0]
    ys = p[1] + offs[:, None] * n_dir[1] + k[None, :] * e_dir[1]
    v = cv2.remap(gray32, xs.astype(np.float32), ys.astype(np.float32),
                  cv2.INTER_LINEAR)
    return v.mean(axis=1)


def cut_scan(gray: np.ndarray, side: str, anchor, scan_us: np.ndarray,
             ppm: float, win_out_mm: float = 3.2, win_in_mm: float = 0.4,
             band: int = 12, min_prom: float = 10.0, plateau_mm: float = 0.35,
             step: float = 0.3):
    """Hybrid cut-edge detector (calibration-prototype port).

    Two mechanisms per scan line, tried in order:
    1. specular ridge - the bare cardboard cut catches light as a bright
       ridge; prominence-gated, sub-pixel by intensity-weighted centroid.
    2. plateau-exit knee - median level L of the border plateau inside the
       cut; first sustained departure from L; sub-pixel via the crossing of
       the local transition slope with L.

    Because the knee keys on the departure from the INTERIOR plateau rather
    than a mid-level threshold between card and mat, a shadow band hugging
    the card edge does not drag the detection outward the way it can for
    step/texture scans (the ~0.5-0.7mm shadow-band artifact).

    anchor: callable u -> v in image coords (e.g. FittedLine.v_at) or a
    constant. The window runs from win_in_mm INSIDE the anchor to win_out_mm
    OUTWARD; the plateau is the window start up to plateau_mm outward of the
    anchor (negative plateau_mm keeps it strictly inside the anchor - use
    that when the anchor is the cut itself rather than an interior line).

    Returns (us, vs, diag); diag.method_counts tallies ridge vs knee.
    """
    if not callable(anchor):
        _a0 = float(anchor)
        anchor = lambda u: _a0
    sign = -1.0 if side in ("left", "top") else 1.0
    horiz = side in ("top", "bottom")
    e_dir = (1.0, 0.0) if horiz else (0.0, 1.0)
    n_dir = (0.0, sign) if horiz else (sign, 0.0)
    g32 = np.ascontiguousarray(gray, dtype=np.float32)
    offs = np.arange(-win_in_mm * ppm, win_out_mm * ppm, step)
    n = len(offs)
    k0 = max(6, int((plateau_mm + win_in_mm) * ppm / step))
    if n < 24 or k0 >= n - 10:
        raise ValueError("cut_scan window too small for plateau/search config")
    H, W = gray.shape
    us, vs = [], []
    diag = ScanDiagnostics()
    for u in scan_us:
        diag.n_attempted += 1
        v0 = float(anchor(u))
        p = (u, v0) if horiz else (v0, u)
        xe, ye = p[0] + offs[-1] * n_dir[0], p[1] + offs[-1] * n_dir[1]
        xb, yb = p[0] + offs[0] * n_dir[0], p[1] + offs[0] * n_dir[1]
        if not (5 < xe < W - 5 and 5 < ye < H - 5
                and 5 < xb < W - 5 and 5 < yb < H - 5):
            diag.note_reject("window_outside_frame")
            continue
        prof = _dir_profile(g32, p, n_dir, offs, band, e_dir)
        off = None
        # --- specular ridge ---
        pk = int(np.argmax(prof[k0:])) + k0
        if 6 <= pk <= n - 7:
            lbase = float(np.median(prof[: max(4, pk - 6)]))
            rbase = float(np.median(prof[min(n - 4, pk + 6):]))
            prom = float(prof[pk]) - max(lbase, rbase)
            if prom >= min_prom:
                half = max(lbase, rbase) + 0.4 * prom
                a = pk
                while a > 0 and prof[a - 1] > half:
                    a -= 1
                b = pk
                while b < n - 1 and prof[b + 1] > half:
                    b += 1
                if (b - a) * step <= 1.2 * ppm and a > 2 and b < n - 3:
                    w = np.clip(prof[a:b + 1] - half, 0, None)
                    if w.sum() > 0:
                        off = float((offs[a:b + 1] * w).sum() / w.sum())
                        diag.note_method("ridge")
        # --- plateau-exit knee ---
        if off is None:
            L = float(np.median(prof[:k0]))
            noise = max(float(np.std(prof[:k0])), 1.0)
            thr = max(6.0, 4.5 * noise)
            dep = np.abs(prof - L) > thr
            # A real cut transition is SHARP; a gradual ramp (e.g. a bright
            # background's glow bleeding into the dark border) also departs
            # from L but over many pixels. Require the local slope to clear
            # several thresholds' worth of level change within ~3px, and
            # re-search past shallow departures (mirrors step_scan's
            # glare-band re-search).
            saw_shallow = False
            i, run = k0, 0
            while i < n:
                run = run + 1 if dep[i] else 0
                if run >= 6:
                    idx = i - 5
                    if idx < 3 or idx > n - 8:
                        break
                    j0, j1 = max(0, idx - 2), min(n, idx + 7)
                    A = np.polyfit(offs[j0:j1], prof[j0:j1], 1)
                    if abs(A[0]) >= 1e-4 and abs(A[0]) * 3.0 >= 2.5 * thr:
                        off = float((L - A[1]) / A[0])
                        if not (offs[max(0, idx - 4)] - 2 * step <= off
                                <= offs[min(n - 1, idx + 8)] + 2 * step):
                            off = float(offs[idx])
                        diag.note_method("knee")
                        break
                    saw_shallow = True
                    run = 0
                i += 1
            if off is None:
                diag.note_reject("shallow_departure_slope" if saw_shallow
                                 else "no_sustained_departure")
                continue
        us.append(float(u))
        vs.append(v0 + sign * off)
        diag.n_ok += 1
    return np.array(us), np.array(vs), diag
