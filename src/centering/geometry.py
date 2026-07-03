"""Geometry & statistics primitives: robust line fitting, rectification,
keystone metrics, homography tilt self-calibration."""
from __future__ import annotations

import math
from typing import Optional

import cv2
import numpy as np

MAD_TO_SIGMA = 1.4826


def robust_polyfit(u: np.ndarray, v: np.ndarray, deg: int = 1,
                   n_iter: int = 4, mad_k: float = 3.25,
                   floor: float = 1.5):
    """Iterative least squares with MAD-based outlier rejection.

    Fits v = poly(u). Returns (coeffs, inlier_mask, rms_inlier_residual).
    Rejection threshold per round: max(mad_k * 1.4826 * MAD, floor).
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    mask = np.isfinite(u) & np.isfinite(v)
    if mask.sum() < deg + 2:
        raise ValueError("not enough points for robust fit")
    if deg == 1 and mask.sum() >= 6:
        # Robust initialization: repeated-median slope (Siegel) followed by
        # LTS C-steps. A clustered same-sign biased tail (e.g. shadow-drifted
        # scan lines at one end of the span) tilts plain LSQ and survives MAD
        # rejection; trimmed refits converge onto the clean majority.
        uu, vv = u[mask], v[mask]
        du = uu[None, :] - uu[:, None]
        dv = vv[None, :] - vv[:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            sl = dv / du
        np.fill_diagonal(sl, np.nan)
        m0 = float(np.nanmedian(np.nanmedian(sl, axis=1)))
        b0 = float(np.median(vv - m0 * uu))
        h = max(int(np.ceil(0.55 * len(uu))), 4)

        def _csteps(m, b, n_steps=15):
            idx = None
            for _ in range(n_steps):
                r = vv - (m * uu + b)
                idx = np.argsort(np.abs(r))[:h]
                c = np.polyfit(uu[idx], vv[idx], 1)
                if abs(c[0] - m) < 1e-12 and abs(c[1] - b) < 1e-9:
                    break
                m, b = float(c[0]), float(c[1])
            r = vv - (m * uu + b)
            loss = float(np.sum(np.sort(r ** 2)[:h]))
            return m, b, idx, loss

        # deterministic multi-start (FAST-LTS style): repeated-median start
        # plus elemental 2-point fits spread across the span
        n_pts = len(uu)
        order = np.argsort(uu)
        starts = [(m0, b0)]
        for k in range(8):
            i = order[int(k * (n_pts - 1) / 7.0)]
            j = order[int((k * (n_pts - 1) // 7 + n_pts // 2) % n_pts)]
            if abs(uu[i] - uu[j]) > 1e-9:
                me = (vv[j] - vv[i]) / (uu[j] - uu[i])
                starts.append((float(me), float(vv[i] - me * uu[i])))
        best = None
        for m_s, b_s in starts:
            cand = _csteps(m_s, b_s)
            if best is None or cand[3] < best[3]:
                best = cand
        m0, b0, idx, _ = best
        r_all = v - (m0 * u + b0)
        s = 1.4826 * float(np.median(np.abs((vv - (m0 * uu + b0))[idx])))
        thr = max(mad_k * s, floor)
        new_mask = mask & (np.abs(r_all) <= thr)
        if new_mask.sum() >= deg + 2:
            mask = new_mask
    for _ in range(n_iter):
        coeffs = np.polyfit(u[mask], v[mask], deg)
        r = v - np.polyval(coeffs, u)
        med = np.median(r[mask])
        mad = np.median(np.abs(r[mask] - med))
        thr = max(mad_k * MAD_TO_SIGMA * mad, floor)
        new_mask = mask & (np.abs(r - med) <= thr)
        if new_mask.sum() < deg + 2 or new_mask.sum() == mask.sum():
            mask = new_mask if new_mask.sum() >= deg + 2 else mask
            break
        mask = new_mask
    coeffs = np.polyfit(u[mask], v[mask], deg)
    r = (v - np.polyval(coeffs, u))[mask]
    rms = float(np.sqrt(np.mean(r**2)))
    return coeffs, mask, rms


class FittedLine:
    """A line fitted as v = m*u + b.

    orientation 'v' (vertical-ish edge): u = y, v = x  -> x = m*y + b
    orientation 'h' (horizontal-ish):    u = x, v = y  -> y = m*x + b
    """

    def __init__(self, orientation: str, m: float, b: float,
                 rms: float = 0.0, n: int = 0, n_rej: int = 0,
                 u_range: tuple[float, float] = (0.0, 0.0),
                 bow_px: Optional[float] = None):
        assert orientation in ("v", "h")
        self.orientation, self.m, self.b = orientation, float(m), float(b)
        self.rms, self.n, self.n_rej = rms, n, n_rej
        self.u_range = u_range
        self.bow_px = bow_px

    @classmethod
    def fit(cls, orientation: str, u, v, **kw) -> "FittedLine":
        coeffs, mask, rms = robust_polyfit(u, v, deg=1, **kw)
        u = np.asarray(u, float)
        # bow: quadratic refit on inliers, sag over the span
        bow = None
        if mask.sum() >= 8:
            q = np.polyfit(u[mask], np.asarray(v, float)[mask], 2)
            span = u[mask].max() - u[mask].min()
            bow = float(abs(q[0]) * span**2 / 4.0)
        return cls(orientation, coeffs[0], coeffs[1], rms=rms,
                   n=int(mask.sum()), n_rej=int((~mask).sum()),
                   u_range=(float(u[mask].min()), float(u[mask].max())),
                   bow_px=bow)

    def v_at(self, u):
        return self.m * np.asarray(u, float) + self.b

    def points(self, n: int = 50) -> np.ndarray:
        """(N,2) x,y image points sampled along the fitted span."""
        u = np.linspace(self.u_range[0], self.u_range[1], n)
        v = self.v_at(u)
        if self.orientation == "v":
            return np.stack([v, u], axis=1)
        return np.stack([u, v], axis=1)

    def angle_from_nominal_deg(self) -> float:
        return math.degrees(math.atan(self.m))


def intersect(v_line: FittedLine, h_line: FittedLine) -> tuple[float, float]:
    """Intersection of a vertical-ish (x=m1*y+b1) and horizontal-ish
    (y=m2*x+b2) line. Returns (x, y)."""
    m1, b1, m2, b2 = v_line.m, v_line.b, h_line.m, h_line.b
    x = (m1 * b2 + b1) / (1.0 - m1 * m2)
    y = m2 * x + b2
    return float(x), float(y)


def corner_quad(left: FittedLine, right: FittedLine,
                top: FittedLine, bottom: FittedLine) -> np.ndarray:
    """Corners TL, TR, BR, BL as (4,2) float array in image px."""
    return np.array([intersect(left, top), intersect(right, top),
                     intersect(right, bottom), intersect(left, bottom)],
                    dtype=np.float64)


def corner_angles(quad: np.ndarray) -> dict:
    """Interior angles (deg) at TL,TR,BR,BL - cut squareness indicator."""
    names = ["TL", "TR", "BR", "BL"]
    out = {}
    for i in range(4):
        p, a, b = quad[i], quad[(i - 1) % 4], quad[(i + 1) % 4]
        v1, v2 = a - p, b - p
        c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        out[names[i]] = math.degrees(math.acos(np.clip(c, -1, 1)))
    return out


def keystone(quad: np.ndarray) -> tuple[float, float]:
    """(width keystone %, height keystone %) from corner quad TL,TR,BR,BL."""
    wt = np.linalg.norm(quad[1] - quad[0])
    wb = np.linalg.norm(quad[2] - quad[3])
    hl = np.linalg.norm(quad[3] - quad[0])
    hr = np.linalg.norm(quad[2] - quad[1])
    kw = (wt - wb) / ((wt + wb) / 2) * 100.0
    kh = (hl - hr) / ((hl + hr) / 2) * 100.0
    return float(kw), float(kh)


def homography_to_card(quad: np.ndarray, card_w_mm: float,
                       card_h_mm: float) -> np.ndarray:
    """H mapping image px -> card plane in mm (TL of card at origin)."""
    dst = np.array([[0, 0], [card_w_mm, 0],
                    [card_w_mm, card_h_mm], [0, card_h_mm]], dtype=np.float32)
    return cv2.getPerspectiveTransform(quad.astype(np.float32), dst)


def transform_points(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, H).reshape(-1, 2)


def self_calibrate_tilt(quad: np.ndarray, card_w_mm: float, card_h_mm: float,
                        image_size: tuple[int, int]):
    """Estimate focal length + tilt from the card->image homography.

    Principal point assumed at image centre; solves the two orthogonality
    constraints for f. Returns (f_px, f_mm_equiv, pitch_deg, yaw_deg,
    total_deg) or None when near-degenerate (fronto-parallel)."""
    W, Hh = image_size
    cx, cy = W / 2.0, Hh / 2.0
    src = np.array([[0, 0], [card_w_mm, 0],
                    [card_w_mm, card_h_mm], [0, card_h_mm]], dtype=np.float32)
    Hcm = cv2.getPerspectiveTransform(src, quad.astype(np.float32))  # card->image
    T = np.array([[1, 0, -cx], [0, 1, -cy], [0, 0, 1]], dtype=np.float64)
    Hc = T @ Hcm
    h1, h2 = Hc[:, 0], Hc[:, 1]
    a1 = h1[0] * h2[0] + h1[1] * h2[1]
    b1 = h1[2] * h2[2]
    a2 = (h1[0]**2 + h1[1]**2) - (h2[0]**2 + h2[1]**2)
    b2 = h1[2]**2 - h2[2]**2
    cands = []
    if b1 != 0 and -a1 / b1 > 0:
        cands.append(-a1 / b1)
    if b2 != 0 and -a2 / b2 > 0:
        cands.append(-a2 / b2)
    if not cands:
        return None
    f2 = float(np.mean(cands))
    f = math.sqrt(f2)
    if not (0.2 * max(W, Hh) < f < 30 * max(W, Hh)):
        return None
    Kinv = np.diag([1.0 / f, 1.0 / f, 1.0])
    r1 = Kinv @ h1
    r2 = Kinv @ h2
    s = math.sqrt(np.linalg.norm(r1) * np.linalg.norm(r2))
    r1, r2 = r1 / np.linalg.norm(r1), r2 / np.linalg.norm(r2)
    r3 = np.cross(r1, r2)
    R = np.stack([r1, r2, r3], axis=1)
    U, _, Vt = np.linalg.svd(R)
    R = U @ Vt
    n = R[:, 2]  # card normal in camera coords
    if n[2] < 0:
        n = -n
    total = math.degrees(math.acos(np.clip(n[2], -1, 1)))
    yaw = math.degrees(math.atan2(n[0], n[2]))
    pitch = math.degrees(math.atan2(n[1], n[2]))
    diag_px = math.hypot(W, Hh)
    f_mm_equiv = f * 43.266 / diag_px
    return f, f_mm_equiv, pitch, yaw, total
