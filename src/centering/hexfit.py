"""Sub-pixel fitting of the ink-cost hexagon (a regular, pointy-top hexagon).

Pure image geometry, no card or game knowledge. Two steps:

1. `find_boundaries` - search a window for the hexagon centre and every
   hexagonal boundary around it (the ring's inner and outer edges, the
   swirl's outline on inkable cards), then refine each boundary from
   sub-pixel edge points on all six sides.
2. `joint_fit` - given which boundaries are which and their physical
   sizes, fit one shared centre and a separate scale for each image axis
   (a regular hexagon of known size is a ruler in both directions).

Side k of the hexagon has outward normal n_k at k*60 degrees, and every
point p on it satisfies n_k . (p - c) = a, where a is the apothem
(centre-to-side distance). Calibrated on the 3,226 official renders
(calibration/hex_anchor_survey.py): 0.1-0.2px fit residuals.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

_ANG = np.deg2rad(np.arange(6) * 60.0)
NORMALS = np.stack([np.cos(_ANG), np.sin(_ANG)], 1)
_TANG = np.stack([-NORMALS[:, 1], NORMALS[:, 0]], 1)
_T_COARSE = np.linspace(-0.45, 0.45, 13)

# a single glyph or artwork edge must not win the coarse search on its own
_GRAD_CLIP = 3000.0
# weakest edge a refinement point may sit on (Scharr units, Lab)
_MIN_EDGE = 200.0


@dataclass
class Boundary:
    cx: float
    cy: float
    apothem: float
    rms: float
    n: int
    strength: float
    # sub-pixel edge points used by the fit: (x, y, side index)
    pts: np.ndarray = field(default=None, repr=False)


def side_apothems(b: Boundary):
    """(vertical-side apothem, slanted-side apothem) of one boundary, px,
    about its own fitted centre."""
    k = b.pts[:, 2].astype(int)
    d = (NORMALS[k, 0] * (b.pts[:, 0] - b.cx)
         + NORMALS[k, 1] * (b.pts[:, 1] - b.cy))
    v = (k % 3) == 0
    return float(np.median(d[v])), float(np.median(d[~v]))


@dataclass
class JointFit:
    cx: float
    cy: float
    ppm_x: float          # image px per mm along x
    ppm_y: float
    rms_mm: float
    n: int
    sigma_cx_px: float
    sigma_cy_px: float
    sigma_ppm_x_rel: float
    sigma_ppm_y_rel: float


def gradients(rgb: np.ndarray):
    """Per-pixel gradient from whichever Lab channel is strongest: the ring
    is gold, white, orange or bronze depending on the card."""
    if rgb.ndim == 2:
        g = rgb.astype(np.float32)
        return (cv2.Scharr(g, cv2.CV_32F, 1, 0) * 0.5,
                cv2.Scharr(g, cv2.CV_32F, 0, 1) * 0.5)
    lab = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2LAB)
    lab = lab.astype(np.float32)
    gx = np.zeros(lab.shape[:2], np.float32)
    gy = gx.copy()
    best = gx.copy()
    for c in range(3):
        x = cv2.Scharr(lab[..., c], cv2.CV_32F, 1, 0)
        y = cv2.Scharr(lab[..., c], cv2.CV_32F, 0, 1)
        m = x * x + y * y
        sel = m > best
        best[sel] = m[sel]
        gx[sel] = x[sel]
        gy[sel] = y[sel]
    return gx, gy


def _score(gx, gy, cx, cy, a):
    """Mean |gradient . normal| over the hexagon perimeter; broadcasts."""
    s = 2 * a / math.sqrt(3)
    tot = 0
    for k in range(6):
        px = cx + NORMALS[k, 0] * a + _TANG[k, 0] * s * _T_COARSE[:, None, None, None]
        py = cy + NORMALS[k, 1] * a + _TANG[k, 1] * s * _T_COARSE[:, None, None, None]
        xi = np.clip(np.rint(px).astype(np.intp), 0, gx.shape[1] - 1)
        yi = np.clip(np.rint(py).astype(np.intp), 0, gx.shape[0] - 1)
        g = np.abs(gx[yi, xi] * NORMALS[k, 0] + gy[yi, xi] * NORMALS[k, 1])
        tot = tot + np.minimum(g, _GRAD_CLIP).mean(0)
    return tot / 6


def _edge_points(gx, gy, cx, cy, a, half, nt=31):
    """Sub-pixel edge points along the six sides of a trial hexagon."""
    s = 2 * a / math.sqrt(3)
    step = 0.25
    offs = np.arange(-half, half + 1e-6, step)
    out = []
    for k in range(6):
        n, t = NORMALS[k], _TANG[k]
        tt = np.linspace(-0.38, 0.38, nt)
        base = (np.array([cx, cy])[None, :] + n[None, :] * a
                + t[None, :] * (s * tt)[:, None])            # (nt,2)
        pts = base[:, None, :] + offs[None, :, None] * n[None, None, :]
        mx = pts[..., 0].astype(np.float32)
        my = pts[..., 1].astype(np.float32)
        g = np.abs(cv2.remap(gx, mx, my, cv2.INTER_LINEAR) * n[0]
                   + cv2.remap(gy, mx, my, cv2.INTER_LINEAR) * n[1])
        j = g.argmax(1)
        for i in range(nt):
            jj = int(j[i])
            if jj == 0 or jj == len(offs) - 1 or g[i, jj] < _MIN_EDGE:
                continue
            den = g[i, jj - 1] - 2 * g[i, jj] + g[i, jj + 1]
            d = 0.5 * (g[i, jj - 1] - g[i, jj + 1]) / den * step if den else 0.0
            p = pts[i, jj] + d * n
            out.append((p[0], p[1], k))
    return np.array(out, np.float64).reshape(-1, 3)


def refine(gx, gy, cx, cy, a, half=None, strength=0.0):
    """Least-squares regular hexagon through sub-pixel edge points, with
    iterative outlier rejection. None when fewer than 5 sides support it."""
    if half is None:
        half = max(3.0, 0.05 * a)
    pts = _edge_points(gx, gy, cx, cy, a, half)
    if len(pts) < 60:
        return None
    k = pts[:, 2].astype(int)
    A = np.column_stack([NORMALS[k, 0], NORMALS[k, 1], np.ones(len(pts))])
    rhs = NORMALS[k, 0] * pts[:, 0] + NORMALS[k, 1] * pts[:, 1]
    keep = np.ones(len(pts), bool)
    for _ in range(3):
        sol, *_ = np.linalg.lstsq(A[keep], rhs[keep], rcond=None)
        res = A @ sol - rhs
        sd = 1.4826 * np.median(np.abs(res[keep])) + 1e-3
        keep = np.abs(res) < max(3 * sd, 0.3)
    if len(set(k[keep].tolist())) < 5 or keep.sum() < 60:
        return None
    rms = float(np.sqrt(np.mean(res[keep] ** 2)))
    return Boundary(float(sol[0]), float(sol[1]), float(sol[2]), rms,
                    int(keep.sum()), float(strength), pts[keep])


def find_boundaries(rgb, centre, win, a_range, max_boundaries=4,
                    work_scale=None):
    """Locate the hexagon near `centre` (px) within +-`win` px, trying
    apothems in `a_range` (px). Returns boundaries sorted by apothem.

    The coarse search runs on a copy downscaled so the inner apothem is
    about 30px (`work_scale` overrides); the refinement runs at full
    resolution."""
    cx0, cy0 = centre
    a_lo, a_hi = a_range
    reach = int(math.ceil(win + a_hi * 1.3 + 8))
    x0 = int(max(0, math.floor(cx0 - reach)))
    y0 = int(max(0, math.floor(cy0 - reach)))
    x1 = int(min(rgb.shape[1], math.ceil(cx0 + reach)))
    y1 = int(min(rgb.shape[0], math.ceil(cy0 + reach)))
    roi = rgb[y0:y1, x0:x1]
    if roi.shape[0] < 16 or roi.shape[1] < 16:
        return []
    sc = work_scale or min(1.0, 30.0 / max(a_lo, 1.0))
    small = cv2.resize(roi, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA) \
        if sc < 1.0 else roi
    sgx, sgy = gradients(cv2.GaussianBlur(small, (3, 3), 0))
    step = 1.0
    cxs = np.arange((cx0 - x0 - win) * sc, (cx0 - x0 + win) * sc + 1e-6, step)
    cys = np.arange((cy0 - y0 - win) * sc, (cy0 - y0 + win) * sc + 1e-6, step)
    As = np.arange(a_lo * sc, a_hi * sc + 1e-6, 0.5)
    best, arg = -1.0, None
    for i0 in range(0, len(cys), 16):          # bounded memory
        m = _score(sgx, sgy, cxs[None, :, None], cys[i0:i0 + 16, None, None],
                   As[None, None, :]).max(2)
        j = np.unravel_index(m.argmax(), m.shape)
        if m[j] > best:
            best, arg = float(m[j]), (cys[i0 + j[0]], cxs[j[1]])
    if arg is None:
        return []
    ccy, ccx = arg[0] / sc, arg[1] / sc
    gx, gy = gradients(roi)
    fstep = max(0.25, 0.5 / sc)
    fx = np.arange(ccx - 2 / sc, ccx + 2 / sc + 1e-6, fstep)
    fy = np.arange(ccy - 2 / sc, ccy + 2 / sc + 1e-6, fstep)
    astep = max(0.25, 0.25 / sc)
    fa = np.arange(a_lo, a_hi * 1.3 + 1e-6, astep)
    m = _score(gx, gy, fx[None, :, None], fy[:, None, None], fa[None, None, :])
    jy, jx = np.unravel_index(m.max(2).argmax(), m.shape[:2])
    prof = m[jy, jx]
    w = max(2, int(round(1.0 / astep)))
    peaks = [i for i in range(w, len(prof) - w)
             if prof[i] == prof[i - w:i + w + 1].max()
             and prof[i] > 0.35 * prof.max()]
    peaks = sorted(peaks, key=lambda i: -prof[i])[:max_boundaries]
    out = []
    for i in peaks:
        b = refine(gx, gy, fx[jx], fy[jy], fa[i], strength=float(prof[i]))
        if b is None:
            continue
        b.cx += x0
        b.cy += y0
        b.pts = b.pts + np.array([x0, y0, 0.0])
        if all(abs(b.apothem - o.apothem) > 0.02 * b.apothem for o in out):
            out.append(b)
    return sorted(out, key=lambda b: b.apothem)


def joint_fit(boundaries, apothems_mm, offsets_mm=None) -> JointFit:
    """One centre and a per-axis scale from boundaries of known size.

    For a point p on side k of a boundary with physical apothem A_k:
        n_kx (x - cx)/ppm_x + n_ky (y - cy)/ppm_y = A_k
    which is linear in U=1/ppm_x, V=1/ppm_y, E=cx*U, F=cy*V.

    `apothems_mm` gives, per boundary, either one apothem or a pair
    (vertical sides, slanted sides): the printed hexagons are not exactly
    regular (inkable: vertical sides 0.6% further out than a regular
    hexagon would put them), and without that the difference would be
    read as the photo being stretched. `offsets_mm` optionally gives each
    boundary's own centre relative to the shared one (a ring can be a
    little thicker on one side)."""
    rows, rhs = [], []
    offsets_mm = offsets_mm or [(0.0, 0.0)] * len(boundaries)
    for b, A, (dx, dy) in zip(boundaries, apothems_mm, offsets_mm):
        a_v, a_s = (A, A) if np.isscalar(A) else A
        k = b.pts[:, 2].astype(int)
        nx, ny = NORMALS[k, 0], NORMALS[k, 1]
        rows.append(np.column_stack([nx * b.pts[:, 0], ny * b.pts[:, 1],
                                     -nx, -ny]))
        rhs.append(np.where((k % 3) == 0, float(a_v), float(a_s))
                   + nx * dx + ny * dy)
    M = np.vstack(rows)
    r = np.concatenate(rhs)
    sol, *_ = np.linalg.lstsq(M, r, rcond=None)
    res = M @ sol - r
    n = len(r)
    dof = max(n - 4, 1)
    s2 = float(res @ res) / dof
    cov = s2 * np.linalg.inv(M.T @ M)
    U, V, E, F = sol
    cx, cy = E / U, F / V
    # delta method for cx = E/U, cy = F/V
    jx = np.array([-E / U ** 2, 0, 1 / U, 0])
    jy = np.array([0, -F / V ** 2, 0, 1 / V])
    return JointFit(
        cx=float(cx), cy=float(cy), ppm_x=float(1 / U), ppm_y=float(1 / V),
        rms_mm=float(math.sqrt(s2)), n=int(n),
        sigma_cx_px=float(math.sqrt(max(jx @ cov @ jx, 0))),
        sigma_cy_px=float(math.sqrt(max(jy @ cov @ jy, 0))),
        sigma_ppm_x_rel=float(math.sqrt(cov[0, 0]) / abs(U)),
        sigma_ppm_y_rel=float(math.sqrt(cov[1, 1]) / abs(V)))
