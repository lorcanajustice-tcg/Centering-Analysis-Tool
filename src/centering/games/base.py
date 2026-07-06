"""Game adapter layer: everything card-game-specific lives here."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Protocol


@dataclass
class FrameLineSpec:
    """Expectations for a printed frame line on a bordered face."""
    min_peak: float = 45.0            # brightness excess over local median
    search_mm: tuple = (0.5, 6.0)     # scan window inward from the card edge
    nominal_border_mm: float = 2.4    # sanity-check target (approximate)


@dataclass
class GameSpec:
    name: str
    card_w_mm: float
    card_h_mm: float
    back_frame: Optional[FrameLineSpec] = None
    # equivalence convention for borderless faces: total (both-sides) margin
    # used to express a print shift as a grading-style ratio
    equiv_margin_lr_mm: float = 4.8
    equiv_margin_tb_mm: float = 6.0
    # official-render crop bias (mm): the render is cropped inside trim
    # ASYMMETRICALLY; a raw render-anchored shift measures
    # true_shift + bias, so the pipeline subtracts these. Per axis:
    # bias = (crop_first_side - crop_second_side) / 2, i.e. x: (left-right)/2,
    # y: (top-bottom)/2, in the shift sign convention (+x toward right edge,
    # +y toward bottom edge). See games/<game>.py for calibration provenance.
    render_crop_bias_mm: dict = field(default_factory=lambda: {
        "x": 0.0, "y": 0.0})
    # 1-sigma systematic uncertainty of that calibration (mm)
    render_crop_bias_unc_mm: dict = field(default_factory=lambda: {
        "x": 0.03, "y": 0.03})
    # calibrated edge-definition uncertainties (px), fixture-derived
    edge_def_px: dict = field(default_factory=lambda: {
        "texture": 3.0, "step": 1.0, "frame_peak": 1.5})
    # physical-plausibility bounds (mm) for the borderless render-span
    # gate: {"x_total": (lo, hi), "y_total": (lo, hi), "side": (lo, hi)}.
    # The cut always lies OUTSIDE the render (crop >= 0) and the per-axis
    # render-to-cut totals are layout-locked constants; a fitted edge
    # violating these cannot be the physical cut (cast shadow / curl /
    # glare). None disables the gate (uncalibrated game).
    render_span_bounds_mm: Optional[dict] = None


class RenderSource(Protocol):
    def get_render(self, card_id: str): ...
