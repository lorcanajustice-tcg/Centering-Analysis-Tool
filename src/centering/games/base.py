"""Game adapter layer: everything card-game-specific lives here."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Protocol


@dataclass
class FrameLineSpec:
    """Expectations for a printed frame line on a bordered face."""
    min_peak: float = 45.0            # brightness excess over local median
    search_mm: tuple = (0.5, 6.0)     # scan window inward from the card edge
    nominal_border_mm: float = 2.4    # sanity-check target (approximate)


@dataclass
class HexLayout:
    """One printed design of the ink-cost hexagon, in official-picture px.

    `rings` are the two edges of the hexagon's ring, inner then outer, each
    (centre dx, centre dy, vertical-side apothem, slanted-side apothem)
    with the centre given relative to `centre_px`. `centre_px` is the mean
    of the two ring-edge centres."""
    name: str
    centre_px: tuple
    rings: tuple
    swirl: bool = False          # the flower-shaped surround (inkable)

    @property
    def mean_apothem_px(self) -> float:
        return sum(0.5 * (r[2] + r[3]) for r in self.rings) / len(self.rings)


@dataclass
class HexAnchorSpec:
    """Where the ink-cost hexagon is printed, for the card-agnostic front
    check (hexanchor.py). All positions are in official-picture pixels and
    turned into millimetres with `render_px_per_mm`."""
    render_size: tuple                       # (width, height) px
    render_px_per_mm: float                  # physical print scale
    render_px_per_mm_rel_unc: float
    layouts: dict                            # name -> HexLayout
    default_layouts: tuple                   # tried when the card is unknown
    # (layout name, set codes) -> centre_px for older printings of a layout
    older_layouts: dict = field(default_factory=dict)
    # per-card survey of every official picture (relative to the game's
    # local card database folder), used when the card is known
    percard_csv: Optional[str] = None
    # search window around the expected centre, mm
    search_mm: float = 2.0
    # hexagon size that disagrees with the card size by more than this is
    # flagged (fraction)
    scale_tol: float = 0.02
    # print scale beyond the calibration: dot gain, ring-width variants
    size_rel_unc: float = 0.004
    # card size spread, mm (1 sigma), per axis
    card_size_sd_mm: dict = field(default_factory=lambda: {"x": 0.07,
                                                           "y": 0.08})


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
    # calibrated edge-definition uncertainties (px), fixture-derived.
    # DETECTOR-domain: how sharply a given scanner resolves any straight
    # boundary. A pixel quantity - it scales with capture resolution.
    edge_def_px: dict = field(default_factory=lambda: {
        "texture": 3.0, "step": 1.0, "colour": 1.0, "frame_peak": 1.5})
    # FACE-domain cut-definition uncertainty, mm per side, per axis:
    # {face: {"x": mm, "y": mm}}, face in {"back", "front"}; a missing face
    # or axis means zero. This is ambiguity about where the physical cut
    # IS, carried by the printed face itself rather than by the scanner -
    # a full-art face cut through artwork presents a rim that a plain
    # card border does not. Physical, so it is a MILLIMETRE quantity and
    # must not scale with resolution (which is why it is not folded into
    # edge_def_px). See games/<game>.py for provenance.
    cut_def_mm: dict = field(default_factory=dict)
    # physical-plausibility bounds (mm) for the borderless render-span
    # gate: {"x_total": (lo, hi), "y_total": (lo, hi), "side": (lo, hi)}.
    # The cut always lies OUTSIDE the render (crop >= 0) and the per-axis
    # render-to-cut totals are layout-locked constants; a fitted edge
    # violating these cannot be the physical cut (cast shadow / curl /
    # glare). None disables the gate (uncalibrated game).
    render_span_bounds_mm: Optional[dict] = None
    # card-agnostic front check from the printed ink-cost hexagon; None
    # when the game has no calibrated layout
    hex_anchor: Optional[HexAnchorSpec] = None

    def edge_def_mm(self, face: str, method: str, axis: str,
                    px_per_mm: float) -> float:
        """1-sigma edge-definition uncertainty (mm) for one fitted CUT edge.

        Composes the detector term (`edge_def_px`, resolution-dependent)
        with this face's physical cut term (`cut_def_mm`, resolution-free).
        `face` is "back" or "front", `axis` is "x" or "y".
        """
        px = self.edge_def_px.get(method) or self.edge_def_px.get("step", 1.0)
        detector_mm = px / max(px_per_mm, 1e-9)
        face_mm = self.cut_def_mm.get(face, {}).get(axis, 0.0)
        return math.hypot(detector_mm, face_mm)


class RenderSource(Protocol):
    def get_render(self, card_id: str): ...
