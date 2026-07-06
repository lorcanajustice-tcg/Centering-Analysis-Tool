"""Coarse card localization.

Brightness segmentation is unreliable on dark backs and glare-lit mats, so the
coarse pass reuses the texture scanner itself: full-span scans from each side
of the frame on a sparse row/column grid. On smooth light backgrounds (white
paper, gray desk) both the mat and the card border are texture-free, so each
side falls back to the polarity-agnostic brightness step scanner. Sides that
fail both are either inferred from the opposite side (when scale is known) or
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


def _consensus(method: str, vs, diag, n_lines: int, mad_limit: float):
    """Median/MAD consensus over one scanner's coarse detections.

    Returns (CoarseSide|None, mad_or_None): the side when consistent, else
    None plus the offending spread (None when there were too few points).
    """
    if diag.n_ok < max(6, n_lines // 3):
        return None, None
    med = float(np.median(vs))
    mad = float(np.median(np.abs(vs - med)))
    if mad > mad_limit:
        return None, mad
    return CoarseSide("ok", med, diag.n_ok, mad, method=method), mad


def coarse_locate(gray: np.ndarray, card_w_mm: float, card_h_mm: float,
                  n_lines: int = 15, mad_limit: float = 30.0):
    """Returns (sides: dict[str, CoarseSide], ppm: float|None)."""
    H, W = gray.shape
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
                                       sustain=30, min_sep=4.0)
        cs, mad_t = _consensus("texture", vs_t, diag_t, n_lines, mad_limit)
        mad_s = None
        if cs is None:
            # smooth light background: no texture signal on either side of
            # the edge; the polarity-agnostic brightness step still works
            _, vs_s, diag_s = step_scan(gray, side, approx, us, so, si,
                                        band=5, min_contrast=25.0)
            cs, mad_s = _consensus("step", vs_s, diag_s, n_lines, mad_limit)
        if cs is not None:
            sides[side] = cs
            continue
        spreads = [m for m in (mad_t, mad_s) if m is not None]
        if spreads:
            sides[side] = CoarseSide(
                "failed", None, max(diag_t.n_ok, diag_s.n_ok), min(spreads),
                f"inconsistent coarse edge detections (spread "
                f"{min(spreads):.0f}px); likely glare bands or background "
                "texture non-uniformity")
        else:
            sides[side] = CoarseSide(
                "failed", None, diag_t.n_ok, 0.0,
                f"insufficient texture/brightness contrast between background "
                f"and card (texture: {diag_t.n_ok}/{diag_t.n_attempted} "
                f"coarse lines usable, {diag_t.summary()}; step: "
                f"{diag_s.n_ok}/{diag_s.n_attempted} usable, "
                f"{diag_s.summary()})")

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
        raise RuntimeError("no card-candidate component found")
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
    return best


# backwards-compatible alias (now polarity-agnostic)
bright_component_bbox = card_component_bbox
