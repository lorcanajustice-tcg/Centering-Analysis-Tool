"""Coarse card localization.

Brightness segmentation is unreliable on dark backs and glare-lit mats, so the
coarse pass reuses the texture scanner itself: full-span scans from each side
of the frame on a sparse row/column grid. Sides that fail coarsely are either
inferred from the opposite side (when scale is known) or reported failed with
the scanner's reject reasons - which become the user-facing refusal text.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .edges import texture_scan


@dataclass
class CoarseSide:
    status: str            # ok | inferred | failed
    pos: Optional[float]   # image coordinate of the edge (x for L/R, y for T/B)
    n_ok: int = 0
    mad_px: float = 0.0
    reason: str = ""


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
        _, vs, diag = texture_scan(gray, side, approx, us, so, si,
                                   sustain=30, min_sep=4.0)
        if diag.n_ok >= max(6, n_lines // 3):
            med = float(np.median(vs))
            mad = float(np.median(np.abs(vs - med)))
            if mad <= mad_limit:
                sides[side] = CoarseSide("ok", med, diag.n_ok, mad)
                continue
            sides[side] = CoarseSide(
                "failed", None, diag.n_ok, mad,
                f"inconsistent coarse edge detections (spread {mad:.0f}px); "
                "likely glare bands or background texture non-uniformity")
        else:
            sides[side] = CoarseSide(
                "failed", None, diag.n_ok, 0.0,
                f"insufficient texture contrast between background and card "
                f"({diag.n_ok}/{diag.n_attempted} coarse lines usable; "
                f"{diag.summary()})")

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


def bright_component_bbox(gray: np.ndarray):
    """Bright-card bbox via centre-vs-frame-ring adaptive threshold (used for
    borderless fronts, where the card is the bright object)."""
    import cv2
    H, W = gray.shape
    c = gray[int(H * .40):int(H * .60), int(W * .40):int(W * .60)]
    ring = np.concatenate([gray[:int(H * .03)].ravel(),
                           gray[-int(H * .03):].ravel(),
                           gray[:, :int(W * .03)].ravel(),
                           gray[:, -int(W * .03):].ravel()])
    thr = 0.5 * (float(np.median(c)) + float(np.median(ring)))
    th = (cv2.GaussianBlur(gray, (9, 9), 0) > thr).astype(np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, np.ones((25, 25), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(th)
    if n < 2:
        raise RuntimeError("no bright component found")
    best, score = None, -1.0
    for i in range(1, n):
        x, y, w, h, a = stats[i]
        fill = a / max(w * h, 1)
        if a / gray.size < 0.10 or fill < 0.7:
            continue
        if a > score:
            best, score = (int(x), int(y), int(x + w), int(y + h)), a
    if best is None:
        raise RuntimeError("no plausible card-shaped bright component")
    return best
