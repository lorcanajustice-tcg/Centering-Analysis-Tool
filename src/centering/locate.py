"""Coarse card localization.

Brightness segmentation is unreliable on dark backs and glare-lit mats, so the
coarse pass reuses the texture scanner itself: full-span scans from each side
of the frame on a sparse row/column grid. On smooth light backgrounds (white
paper, gray desk) both the mat and the card border are texture-free, so each
side falls back to the polarity-agnostic brightness step scanner. Sides that
fail both fall back to the tight-crop path (background level taken from the
image border, for scans and pre-cropped exports where the card nearly fills
the frame), are inferred from the opposite side (when scale is known), or are
reported failed with the scanners' reject reasons - which become the
user-facing refusal text.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .edges import step_scan, texture_scan


@dataclass
class CoarseSide:
    status: str            # ok | inferred | failed
    pos: Optional[float]   # image coordinate of the edge (x for L/R, y for T/B)
    n_ok: int = 0
    mad_px: float = 0.0
    reason: str = ""
    method: str = ""       # texture | step (scanner that produced pos)
    slop_mm: Optional[float] = None
    """How far the fine scan should search either side of `pos`, when this
    seed knows better than the pipeline default. The tight-crop path reads
    the same border feature the fine scanners will, so its seed is good to
    a fraction of a millimetre and a narrow window keeps the scanners'
    background/card level estimates from reaching into distant artwork."""


def _largest_cluster(v: np.ndarray, width: float):
    """Indices (i, j) of the largest run in sorted v spanning <= width."""
    best_i, best_j = 0, -1
    j = 0
    for i in range(len(v)):
        if j < i:
            j = i
        while j + 1 < len(v) and v[j + 1] - v[i] <= width:
            j += 1
        if j - i > best_j - best_i:
            best_i, best_j = i, j
    return best_i, best_j


def _consensus(method: str, vs, diag, n_lines: int, mad_limit: float,
               cluster_ok: bool = False):
    """Largest-cluster consensus over one scanner's coarse detections.

    A global median/MAD fails when a minority of scan lines latch onto a
    shadow boundary or glare band whose position varies along the edge
    (each contaminated line reports a different position, blowing up the
    MAD even though the true-edge lines agree). Instead: take the largest
    cluster of detections spanning <= 2*mad_limit. It must be big enough
    AND clearly dominate any second cluster elsewhere - two comparable
    consistent clusters (e.g. a straight shadow edge parallel to the card
    edge) are ambiguous and the side is refused rather than guessed.

    Returns (CoarseSide|None, spread_or_None) as before.
    """
    need = max(6, n_lines // 3)
    if diag.n_ok < need:
        return None, None
    v = np.sort(np.asarray(vs, dtype=float))
    med = float(np.median(v))
    mad_all = float(np.median(np.abs(v - med)))
    if not cluster_ok:
        # strict global consensus (dark mats, and the texture scanner
        # everywhere): glare bands produce internally-consistent minority
        # clusters that MUST stay refused (see test_elsa_back).
        if mad_all > mad_limit:
            return None, mad_all
        return CoarseSide("ok", med, diag.n_ok, mad_all, method=method), mad_all
    i, j = _largest_cluster(v, 2.0 * mad_limit)
    cluster = v[i:j + 1]
    # Supermajority rule: the cluster must hold >=60% of the detections
    # (and >= the absolute minimum); a minority cluster - however
    # internally consistent - is refused rather than guessed.
    if len(cluster) < max(need, int(np.ceil(0.6 * diag.n_ok))):
        return None, mad_all
    cmed = float(np.median(cluster))
    cmad = float(np.median(np.abs(cluster - cmed)))
    return CoarseSide("ok", cmed, len(cluster), cmad, method=method), cmad


# --------------------------------------------------------------------------
# tight-crop localization (card nearly fills the frame)
# --------------------------------------------------------------------------
# The per-side coarse scanners sample their "background" level from a wide
# window outside the nominal edge (30% of the frame). When the card fills
# the frame - flatbed scans, grading-report exports, or a photo the user
# already cropped tight - that window is mostly card, so the contrast test
# fails on every side and the whole photo is refused even though the edge
# itself is perfectly crisp. This path takes the background level from the
# outermost few pixels of the image instead and walks inward to the first
# sustained departure from it. It only produces a *coarse* seed: the
# sub-pixel fits, tilt correction and refusal gates downstream are
# unchanged.

TIGHT_MARGIN_FRAC = 0.12     # search depth as a fraction of the dimension
TIGHT_ASPECT_TOL = 0.03      # box aspect must match the card to 3%
TIGHT_SLOP_MM = 0.6          # fine-scan window around a tight-crop seed


def _edge_band(gray: np.ndarray, side: str, depth: int, lo: float, hi: float):
    """[line, sample] block along one image border, sample ordered inward."""
    H, W = gray.shape
    a, b = int(lo), int(hi)
    if side == "left":
        return gray[a:b, :depth]
    if side == "right":
        return gray[a:b, W - depth:][:, ::-1]
    if side == "top":
        return gray[:depth, a:b].T
    return gray[H - depth:, a:b][::-1].T


def _margin_depth(block: np.ndarray, strip: int, sustain: int):
    """First sample at which a uniform border band gives way to the card.

    Two signals, either of which ends the margin:

    * **level** - the card sits at a different brightness than the border.
      This is what carries a dark card back against a light surround.
    * **along-edge variation** - a margin is *uniform along its edge*
      (a mat photographed out of focus, a scanner platen, the flat colour
      a grading service pads its crops with), while card artwork varies
      from one scan line to the next. This carries the cases where the
      artwork happens to sit at the border's brightness, where a level
      test alone walks deep into the card.

    Returns the sample index, or None.
    """
    lev = np.median(block, axis=0)
    var = np.median(np.abs(block - lev[None, :]), axis=0)
    bg = float(np.median(lev[:strip]))
    bg_lev_mad = float(np.median(np.abs(lev[:strip] - bg)))
    bg_var = float(np.median(var[:strip]))
    thr_lev = max(10.0, 6.0 * bg_lev_mad)
    thr_var = max(4.0, 3.0 * bg_var + 3.0)
    depart = (np.abs(lev - bg) > thr_lev) | (var > thr_var)
    run = 0
    for k, d in enumerate(depart):
        run = run + 1 if d else 0
        if run >= sustain:
            idx = k - sustain + 1
            return idx if idx > 0 else None
    return None


def tight_crop_locate(gray: np.ndarray, card_w_mm: float, card_h_mm: float,
                      thirds_tol_frac: float = 0.006):
    """Coarse localization for a card that nearly fills the frame.

    Returns (sides, ppm) on success, or None when the geometry does not
    check out. Two gates keep this from inventing a card in an ordinary
    photo: each edge is located independently in three sections along its
    length and they must agree, and the resulting box must have the
    card's aspect ratio to TIGHT_ASPECT_TOL. A box that is not a card
    practically never clears both.
    """
    H, W = gray.shape
    strip = max(3, int(round(0.002 * min(H, W))))
    sustain = max(3, int(round(0.0015 * min(H, W))))
    out = {}
    for side in ("left", "right", "top", "bottom"):
        dim = W if side in ("left", "right") else H
        along = H if side in ("left", "right") else W
        depth = int(round(TIGHT_MARGIN_FRAC * dim))
        if depth < strip + sustain + 4:
            return None
        lo, hi = 0.15 * along, 0.85 * along
        idx = _margin_depth(_edge_band(gray, side, depth, lo, hi),
                            strip, sustain)
        if idx is None:
            return None
        # sections along the edge must agree: a real card edge is straight
        # and parallel to the crop, a chance uniform patch is not
        thirds = []
        for k in range(3):
            t0 = lo + k * (hi - lo) / 3.0
            t1 = lo + (k + 1) * (hi - lo) / 3.0
            ti = _margin_depth(_edge_band(gray, side, depth, t0, t1),
                               strip, sustain)
            if ti is None:
                return None
            thirds.append(ti)
        if max(thirds) - min(thirds) > max(4.0, thirds_tol_frac * dim):
            return None
        pos = float(idx) if side in ("left", "top") else float(dim - idx)
        out[side] = CoarseSide(
            "ok", pos, len(thirds), float(max(thirds) - min(thirds)),
            reason="found from the edge of the image, because the card "
                   "nearly fills it",
            method="step", slop_mm=TIGHT_SLOP_MM)

    w_px = out["right"].pos - out["left"].pos
    h_px = out["bottom"].pos - out["top"].pos
    if w_px <= 0 or h_px <= 0:
        return None
    nominal = card_w_mm / card_h_mm
    if abs((w_px / h_px) / nominal - 1.0) > TIGHT_ASPECT_TOL:
        return None
    return out, w_px / card_w_mm


def coarse_locate(gray: np.ndarray, card_w_mm: float, card_h_mm: float,
                  n_lines: int = 15, mad_limit: float = 30.0):
    """Returns (sides: dict[str, CoarseSide], ppm: float|None)."""
    H, W = gray.shape
    # The largest-cluster (shadow-tolerant) consensus is only sound on
    # light backgrounds (white-paper protocol), where hand/phone shadows
    # contaminate a minority of scan lines but matte paper cannot produce
    # glare bands. On dark mats keep the strict global consensus.
    ring = np.concatenate([gray[:int(H * .03)].ravel(),
                           gray[-int(H * .03):].ravel(),
                           gray[:, :int(W * .03)].ravel(),
                           gray[:, -int(W * .03):].ravel()])
    light_bg = float(np.median(ring)) > 128.0
    rows = np.linspace(0.22 * H, 0.78 * H, n_lines)
    cols = np.linspace(0.22 * W, 0.78 * W, n_lines)
    cfg = {
        "left":   (W * 0.30, W * 0.30, W * 0.25, rows),
        "right":  (W * 0.70, W * 0.30, W * 0.25, rows),
        "top":    (H * 0.30, H * 0.30, H * 0.25, cols),
        "bottom": (H * 0.70, H * 0.30, H * 0.25, cols),
    }
    sides: dict[str, CoarseSide] = {}
    for side, (approx, so, si, us) in cfg.items():
        _, vs_t, diag_t = texture_scan(gray, side, approx, us, so, si,
                                       sustain=30, min_sep=4.0,
                                       allow_inverted=False)
        cs, mad_t = _consensus("texture", vs_t, diag_t, n_lines, mad_limit)
        mad_s = None
        if cs is None:
            # smooth light background: no texture signal on either side of
            # the edge; the polarity-agnostic brightness step still works
            _, vs_s, diag_s = step_scan(gray, side, approx, us, so, si,
                                        band=5, min_contrast=25.0)
            cs, mad_s = _consensus("step", vs_s, diag_s, n_lines, mad_limit,
                                   cluster_ok=light_bg)
        if cs is not None:
            sides[side] = cs
            continue
        spreads = [m for m in (mad_t, mad_s) if m is not None]
        if spreads:
            sides[side] = CoarseSide(
                "failed", None, max(diag_t.n_ok, diag_s.n_ok), min(spreads),
                f"the readings along this edge did not agree with each "
                f"other - they were spread over {min(spreads):.0f} pixels, "
                "which usually means a band of glare, or a background that "
                "changes too much along the edge")
        else:
            sides[side] = CoarseSide(
                "failed", None, diag_t.n_ok, 0.0,
                f"the background and the card look too alike along this "
                f"edge. Looking for a change in surface: {diag_t.n_ok} of "
                f"{diag_t.n_attempted} readings usable ({diag_t.summary()}). "
                f"Looking for a change in brightness: {diag_s.n_ok} of "
                f"{diag_s.n_attempted} usable ({diag_s.summary()})")

    # Tight crop / scan fallback: a side with (almost) no background
    # outside the card gives the standard scanners nothing to measure
    # contrast against. Retry from the image-border level; the aspect
    # gate inside keeps this from inventing a box on ordinary photos.
    if any(s.status == "failed" for s in sides.values()):
        tight = tight_crop_locate(gray, card_w_mm, card_h_mm)
        if tight is not None:
            tsides, _ = tight
            for side, cs in sides.items():
                if cs.status == "failed":
                    sides[side] = tsides[side]

    ppm = None
    if sides["left"].status == "ok" and sides["right"].status == "ok":
        ppm = (sides["right"].pos - sides["left"].pos) / card_w_mm
    elif sides["top"].status == "ok" and sides["bottom"].status == "ok":
        ppm = (sides["bottom"].pos - sides["top"].pos) / card_h_mm

    if ppm is not None and ppm > 0:
        pairs = {"top": ("bottom", -card_h_mm), "bottom": ("top", card_h_mm),
                 "left": ("right", -card_w_mm), "right": ("left", card_w_mm)}
        for side, (opp, d_mm) in pairs.items():
            if sides[side].status == "failed" and sides[opp].status == "ok":
                est = sides[opp].pos + d_mm * ppm
                lim = H if side in ("top", "bottom") else W
                if -0.1 * lim < est < 1.1 * lim:
                    sides[side] = CoarseSide("inferred", est, 0, 0.0,
                                             sides[side].reason)
    return sides, ppm


def background_uniformity(gray: np.ndarray, patch: int = 220) -> dict:
    """Median brightness of the four frame-corner patches (assumed mat)."""
    return {
        "top_left": float(np.median(gray[:patch, :patch])),
        "top_right": float(np.median(gray[:patch, -patch:])),
        "bottom_left": float(np.median(gray[-patch:, :patch])),
        "bottom_right": float(np.median(gray[-patch:, -patch:])),
    }


def card_component_bbox(gray: np.ndarray):
    """Card bbox via centre-vs-frame-ring adaptive threshold.

    Polarity-agnostic: the card may be the bright object on a dark mat or
    the dark object on a light background (frame ring assumed to be mat).
    On the historical bright-on-dark path the behaviour is unchanged; on
    the dark-card path a morphological CLOSE first fills bright artwork
    holes so the fill-ratio gate still sees a solid card."""
    import cv2
    H, W = gray.shape
    c = gray[int(H * .40):int(H * .60), int(W * .40):int(W * .60)]
    ring = np.concatenate([gray[:int(H * .03)].ravel(),
                           gray[-int(H * .03):].ravel(),
                           gray[:, :int(W * .03)].ravel(),
                           gray[:, -int(W * .03):].ravel()])
    c_med, r_med = float(np.median(c)), float(np.median(ring))
    thr = 0.5 * (c_med + r_med)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    bright_card = c_med >= r_med
    th = ((blur > thr) if bright_card else (blur < thr)).astype(np.uint8)
    if not bright_card:
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((25, 25), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(th)
    if n < 2:
        raise RuntimeError("nothing in this photo is shaped like a card")
    best, score = None, -1.0
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        fill = a / max(w * h, 1)
        if a / gray.size < 0.10 or fill < 0.7:
            continue
        if a > score:
            best, score = (int(x), int(y), int(x + w), int(y + h)), a
    if best is None:
        raise RuntimeError("no plausible card-shaped component "
                           f"(centre {c_med:.0f} vs ring {r_med:.0f})")
    # A real card never touches the photo frame (capture protocol requires
    # >=5% clearance); a frame-touching component means the segmentation
    # leaked into a shadow band or the card is cut off. Refuse so the
    # caller falls back to per-side coarse scans.
    bx0, by0, bx1, by1 = best
    tol = max(2, int(0.005 * min(H, W)))
    if bx0 <= tol or by0 <= tol or bx1 >= W - tol or by1 >= H - tol:
        raise RuntimeError(
            f"card-candidate component touches the photo frame (bbox "
            f"{best}); segmentation leaked into a shadow band or the card "
            "is not fully inside the frame")
    return best


# backwards-compatible alias (now polarity-agnostic)
bright_component_bbox = card_component_bbox
