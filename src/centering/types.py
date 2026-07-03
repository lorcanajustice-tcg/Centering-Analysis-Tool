"""Typed result objects. Every reported number is a Measurement, never a bare float."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

SCHEMA_VERSION = "1.0"


def _r(x: Optional[float], nd: int = 4) -> Optional[float]:
    return None if x is None else round(float(x), nd)


@dataclass
class Uncertainty:
    """Error budget, kept as separate terms. Units match the parent Measurement."""
    statistical: float = 0.0
    perspective: float = 0.0
    edge_definition: float = 0.0

    @property
    def total(self) -> float:
        return math.sqrt(
            self.statistical**2 + self.perspective**2 + self.edge_definition**2
        )

    def to_dict(self) -> dict:
        return {
            "statistical": _r(self.statistical),
            "perspective": _r(self.perspective),
            "edge_definition": _r(self.edge_definition),
            "total": _r(self.total),
        }


@dataclass
class Measurement:
    value: Optional[float]
    unit: str
    uncertainty: Optional[Uncertainty] = None
    status: str = "measured"  # "measured" | "refused"
    refusal_reason: Optional[str] = None

    @classmethod
    def refused(cls, unit: str, reason: str) -> "Measurement":
        return cls(value=None, unit=unit, uncertainty=None,
                   status="refused", refusal_reason=reason)

    def to_dict(self) -> dict:
        d: dict[str, Any] = {"value": _r(self.value), "unit": self.unit,
                             "status": self.status}
        if self.uncertainty is not None:
            d["uncertainty"] = self.uncertainty.to_dict()
        if self.refusal_reason:
            d["refusal_reason"] = self.refusal_reason
        return d


@dataclass
class Ratio:
    """Centering ratio for one axis. Convention: first value is the Left (or Top)
    border's share of the two borders' total, in percent; larger = wider border
    on that side. display uses standard grading format rounded to integers."""
    axis: str  # "LR" | "TB"
    first_pct: Optional[float] = None
    uncertainty_pts: Optional[Uncertainty] = None
    status: str = "measured"
    refusal_reason: Optional[str] = None

    @classmethod
    def refused(cls, axis: str, reason: str) -> "Ratio":
        return cls(axis=axis, status="refused", refusal_reason=reason)

    @property
    def display(self) -> Optional[str]:
        if self.first_pct is None:
            return None
        a = int(math.floor(self.first_pct + 0.5))  # grading-style half-up
        return f"{a}/{100 - a}"

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "axis": self.axis,
            "first_pct": _r(self.first_pct, 1),
            "second_pct": _r(None if self.first_pct is None else 100 - self.first_pct, 1),
            "display": self.display,
            "convention": "first=Left share" if self.axis == "LR" else "first=Top share",
            "status": self.status,
        }
        if self.uncertainty_pts is not None:
            d["uncertainty_pts"] = self.uncertainty_pts.to_dict()
        if self.refusal_reason:
            d["refusal_reason"] = self.refusal_reason
        return d


@dataclass
class EdgeFitReport:
    edge: str            # left|right|top|bottom (physical card edge) or frame_left|...
    method: str          # texture|step|frame_peak
    n_points: int = 0
    n_rejected: int = 0
    rms_residual_px: Optional[float] = None
    angle_deg: Optional[float] = None      # deviation from nominal orientation
    bow_px: Optional[float] = None         # quadratic sag over span (curl indicator)
    status: str = "ok"                     # ok|flagged|refused
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"edge": self.edge, "method": self.method,
                "n_points": self.n_points, "n_rejected": self.n_rejected,
                "rms_residual_px": _r(self.rms_residual_px, 2),
                "angle_deg": _r(self.angle_deg, 3), "bow_px": _r(self.bow_px, 2),
                "status": self.status, "notes": list(self.notes)}


@dataclass
class TiltReport:
    keystone_w_pct: Optional[float] = None   # (top width - bottom width)/mean *100
    keystone_h_pct: Optional[float] = None
    pitch_deg: Optional[float] = None
    yaw_deg: Optional[float] = None
    total_deg: Optional[float] = None
    focal_mm_equiv: Optional[float] = None   # 35mm-equivalent, self-calibrated
    corrected: bool = True                   # homography rectification applied
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"keystone_w_pct": _r(self.keystone_w_pct, 3),
                "keystone_h_pct": _r(self.keystone_h_pct, 3),
                "pitch_deg": _r(self.pitch_deg, 2), "yaw_deg": _r(self.yaw_deg, 2),
                "total_deg": _r(self.total_deg, 2),
                "focal_mm_equiv": _r(self.focal_mm_equiv, 1),
                "corrected": self.corrected, "notes": list(self.notes)}


@dataclass
class QAFlag:
    code: str
    message: str
    severity: str = "warning"  # info|warning

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message,
                "severity": self.severity}


@dataclass
class InputReport:
    photo: str
    sha256: str
    width: int
    height: int
    exif_transposed: bool
    px_per_mm: Optional[float] = None

    def to_dict(self) -> dict:
        return {"photo": self.photo, "sha256": self.sha256,
                "width": self.width, "height": self.height,
                "exif_transposed": self.exif_transposed,
                "px_per_mm": _r(self.px_per_mm, 2)}


@dataclass
class RenderMatchReport:
    source: str
    url: Optional[str]
    render_size: tuple[int, int]
    n_inliers: int
    median_reproj_px: float
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"source": self.source, "url": self.url,
                "render_size": list(self.render_size),
                "n_inliers": self.n_inliers,
                "median_reproj_px": _r(self.median_reproj_px, 2),
                "notes": list(self.notes)}


@dataclass
class FaceResult:
    """Common envelope for one analyzed face."""
    kind: str                       # "back" | "borderless"
    game: str
    input: InputReport = None
    tilt: TiltReport = None
    edge_fits: list[EdgeFitReport] = field(default_factory=list)
    qa: list[QAFlag] = field(default_factory=list)
    overlay: Optional[str] = None
    # geometry QA
    aspect_ratio_measured: Optional[float] = None   # H/W after rectification input
    corner_angles_deg: Optional[dict] = None        # informational cut-squareness

    def _base_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": self.kind,
            "game": self.game,
            "input": self.input.to_dict() if self.input else None,
            "tilt": self.tilt.to_dict() if self.tilt else None,
            "edge_fits": [e.to_dict() for e in self.edge_fits],
            "aspect_ratio_measured": _r(self.aspect_ratio_measured, 4),
            "corner_angles_deg": (
                {k: _r(v, 2) for k, v in self.corner_angles_deg.items()}
                if self.corner_angles_deg else None),
            "qa": [q.to_dict() for q in self.qa],
            "overlay": self.overlay,
        }


@dataclass
class BackResult(FaceResult):
    borders_mm: dict = field(default_factory=dict)  # L/R/T/B -> Measurement
    ratio_lr: Ratio = None
    ratio_tb: Ratio = None

    def to_dict(self) -> dict:
        d = self._base_dict()
        d["result"] = {
            "borders_mm": {k: m.to_dict() for k, m in self.borders_mm.items()},
            "ratio_lr": self.ratio_lr.to_dict() if self.ratio_lr else None,
            "ratio_tb": self.ratio_tb.to_dict() if self.ratio_tb else None,
        }
        return d


@dataclass
class BorderlessResult(FaceResult):
    render: RenderMatchReport = None
    shift_mm: dict = field(default_factory=dict)    # "x","y" -> Measurement
    shift_convention: str = ("positive x = print displaced toward right card edge; "
                             "positive y = toward bottom edge")
    equivalent_ratio_lr: Ratio = None
    equivalent_ratio_tb: Ratio = None
    per_side_offsets_mm: Optional[dict] = None      # relative only; render-crop caveat

    def to_dict(self) -> dict:
        d = self._base_dict()
        d["result"] = {
            "render": self.render.to_dict() if self.render else None,
            "shift_mm": {k: m.to_dict() for k, m in self.shift_mm.items()},
            "shift_convention": self.shift_convention,
            "equivalent_ratio_lr": (self.equivalent_ratio_lr.to_dict()
                                    if self.equivalent_ratio_lr else None),
            "equivalent_ratio_tb": (self.equivalent_ratio_tb.to_dict()
                                    if self.equivalent_ratio_tb else None),
            "per_side_offsets_mm": self.per_side_offsets_mm,
            "assumption": ("official render crop is layout-locked but NOT "
                           "symmetric about print centre; a calibrated "
                           "per-axis crop bias is subtracted from the raw "
                           "shift (see GameSpec.render_crop_bias_mm). "
                           "per_side_offsets_mm remain raw render-relative "
                           "values, not absolute borders"),
        }
        return d


@dataclass
class CardResult:
    game: str
    back: Optional[BackResult] = None
    front: Optional[BorderlessResult] = None
    registration_mm: Optional[dict] = None   # x,y -> Measurement (front vs back cut agreement)
    mirror_consistency: Optional[str] = None
    qa: list[QAFlag] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "kind": "card",
            "game": self.game,
            "back": self.back.to_dict() if self.back else None,
            "front": self.front.to_dict() if self.front else None,
            "registration_mm": ({k: m.to_dict() for k, m in self.registration_mm.items()}
                                if self.registration_mm else None),
            "mirror_consistency": self.mirror_consistency,
            "qa": [q.to_dict() for q in self.qa],
        }
