"""Verification overlay rendering: every measurement must be visually auditable."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

C_EDGE = (0, 255, 255)     # fitted physical edge: yellow (BGR)
C_FRAME = (255, 128, 0)    # fitted frame line: blue-ish
C_PT = (0, 220, 0)         # accepted scan point: green
C_TXT = (255, 255, 255)


class Overlay:
    def __init__(self, rgb: np.ndarray):
        self.img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
        s = max(self.img.shape[:2]) / 4000.0
        self.lw = max(2, int(round(3 * s)))
        self.pr = max(3, int(round(4 * s)))
        self.fs = 1.2 * s

    def points(self, pts_xy, color=C_PT):
        for x, y in np.asarray(pts_xy).reshape(-1, 2):
            cv2.circle(self.img, (int(round(x)), int(round(y))), self.pr,
                       color, -1)

    def line(self, fitted, color=C_EDGE, extend: float = 0.05):
        u0, u1 = fitted.u_range
        span = u1 - u0
        u0, u1 = u0 - extend * span, u1 + extend * span
        p0v, p1v = fitted.v_at(u0), fitted.v_at(u1)
        if fitted.orientation == "v":
            p0, p1 = (p0v, u0), (p1v, u1)
        else:
            p0, p1 = (u0, p0v), (u1, p1v)
        cv2.line(self.img, tuple(int(round(c)) for c in p0),
                 tuple(int(round(c)) for c in p1), color, self.lw)

    def label(self, text: str, xy, color=C_TXT):
        x, y = int(round(xy[0])), int(round(xy[1]))
        cv2.putText(self.img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    self.fs, (0, 0, 0), int(self.lw * 2.5), cv2.LINE_AA)
        cv2.putText(self.img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    self.fs, color, self.lw, cv2.LINE_AA)

    def banner(self, lines: list):
        y = int(60 * self.fs)
        for t in lines:
            self.label(t, (30, y))
            y += int(55 * self.fs)

    def save(self, path: str | Path) -> str:
        path = str(path)
        cv2.imwrite(path, self.img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return path
