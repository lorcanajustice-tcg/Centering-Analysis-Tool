"""Disney Lorcana adapter."""
from __future__ import annotations

from .base import FrameLineSpec, GameSpec

LORCANA = GameSpec(
    name="lorcana",
    card_w_mm=63.5,
    card_h_mm=88.9,
    back_frame=FrameLineSpec(min_peak=45.0, search_mm=(0.5, 6.0),
                             nominal_border_mm=2.4),
    # Equivalence-margin convention totals (cut-to-frame, both sides),
    # re-derived 2026-07-06 from FIVE white-background backs (12:147,
    # 12:24, 12:133, 12:96, 12-54-P3): L+R 4.775 +- 0.017 (sd) mm --
    # remarkably manufacture-constant -- and T+B 4.305 +- 0.078 (sd) mm.
    # (The former T+B figure of 4.6 came from the 2026-07-03 dark-mat
    # shoot, whose horizontal edges were shadow-inflated outward.)
    equiv_margin_lr_mm=4.78,
    equiv_margin_tb_mm=4.31,
    # Render-crop bias, RE-CALIBRATED 2026-07-06 under the improved
    # protocol (white paper, diffuse light, polarity-agnostic detectors;
    # photos IMG_6397/98 + IMG_6403-6410, results in
    # calibration/reshoot_2026_07_06.json). Estimator: per-pair
    # front_raw_shift minus back-frame-derived shift (vertical-axis flip
    # convention CONFIRMED from two strong-signal pairs: T/B does not
    # mirror, L/R does), which cancels each card's die-cut offset.
    # y = -0.08 +- 0.07 (sem) mm over 5 pairs, card scatter +-0.16
    # (front-back print registration; one pair reached 0.43mm in x) --
    # i.e. CONSISTENT WITH ZERO. The 2026-07-03 value (+0.18 +- 0.06,
    # four dark-mat pairs) is superseded: its positive offset matches the
    # shadow-band mechanism (directional light displacing horizontal-edge
    # scans outward, worst at the top), the same artifact that excluded
    # the Simba pair from that calibration.
    # Full-art check: the 12-54-P3 (enchanted-layout) pair reads -0.26,
    # within ~1.1 sigma of the mean given the registration scatter -> no
    # evidence for a separate full-art constant; single constant retained
    # (as the anchor survey over all 3211 renders predicts: footer emblem
    # y=1899-1900 +-1px on bordered AND full-art layouts).
    # x kept at 0: the render x-crop is anchor-locked symmetric (60/60px
    # std frame); the pair estimator read -0.13 +- 0.09, attributed to
    # registration noise (1.5 sigma, and a true x bias is excluded by the
    # anchors). NOTE: this constant operationally includes any off-centre
    # of the printed back frame (inseparable in the pair estimator; see
    # calibration/NOTES.md "2026-07-06 reshoot").
    render_crop_bias_mm={"x": 0.0, "y": -0.08},
    render_crop_bias_unc_mm={"x": 0.05, "y": 0.10},
    # Render-span gate bounds. Empirical over the seven clean 2026-07-06
    # white/kraft captures (five reshoot fronts + Gadget white/kraft +
    # Ursula 3/D23): x totals 0.73-0.90mm, y totals 1.21-1.97mm, max
    # clean single side 1.33mm; margins ~0.3mm. Motivating failure: a
    # hard cast shadow hugging a dark full-art top edge fakes a sharp
    # "cut" (IMG_6416: top edge measured +2.24mm outside the render,
    # y-total 2.77mm) that per-line edge QA cannot distinguish locally.
    render_span_bounds_mm={"x_total": (0.50, 1.15),
                           "y_total": (1.05, 2.15),
                           "side": (-0.10, 1.90)},
)

ALLCARDS_URL = "https://lorcanajson.org/files/current/en/allCards.json"


import re
from typing import Optional

import cv2
import numpy as np

from ..cache import DiskCache

ALLCARDS_ZIP_URL = "https://lorcanajson.org/files/current/en/allCards.json.zip"


class CardNotFound(ValueError):
    pass


class LorcanaRenderSource:
    """Official Ravensburger renders located via lorcanajson.org.

    lorcanajson includes promos that other APIs miss. Accepted card ids:
    - "6/C2"           number / promo grouping
    - "7:69"           setCode : number
    - "Elsa - Ice Maker" (name or "name - version" substring, unique match)
    """

    def __init__(self, cache: Optional[DiskCache] = None):
        self.cache = cache or DiskCache()
        self._cards = None

    def _load(self):
        if self._cards is None:
            j = self.cache.fetch_json_maybe_zipped(ALLCARDS_ZIP_URL)
            self._cards = j["cards"]
        return self._cards

    def resolve(self, card_id: str) -> dict:
        cards = self._load()
        m = re.fullmatch(r"(\d+)([A-Za-z])?\s*/\s*([A-Za-z][A-Za-z0-9]*)",
                         card_id.strip())
        if m:
            num, var, grp = int(m.group(1)), m.group(2), m.group(3).upper()
            hits = [c for c in cards if c.get("number") == num
                    and (c.get("promoGrouping") or "").upper() == grp
                    and (var is None
                         or (c.get("variant") or "").upper() == var.upper())]
            if len(hits) == 1:
                return hits[0]
            raise CardNotFound(
                f"{card_id}: {len(hits)} matches for promo lookup"
                + (f" (variants: {[c.get('fullIdentifier') for c in hits[:5]]})"
                   if hits else ""))
        m = re.fullmatch(r"(\w+)\s*:\s*(\d+)", card_id.strip())
        if m:
            sc, num = m.group(1), int(m.group(2))
            hits = [c for c in cards if str(c.get("setCode")) == sc
                    and c.get("number") == num and not c.get("promoGrouping")]
            if len(hits) == 1:
                return hits[0]
            raise CardNotFound(f"{card_id}: {len(hits)} matches for set:number lookup")
        q = card_id.strip().lower()
        hits = [c for c in cards
                if q in f"{c.get('name','')} - {c.get('version','')}".lower()]
        if len(hits) == 1:
            return hits[0]
        raise CardNotFound(
            f"{card_id!r}: {len(hits)} name matches"
            + (f" (e.g. {[c.get('fullIdentifier') for c in hits[:5]]})" if hits else ""))

    def get_render(self, card_id: str):
        """Returns (render_gray float32, render_rgb uint8, url, card_dict)."""
        card = self.resolve(card_id)
        url = (card.get("images") or {}).get("full")
        if not url:
            raise CardNotFound(f"{card_id}: no official render URL in database")
        p = self.cache.fetch(url, suffix=".img")
        buf = np.frombuffer(p.read_bytes(), np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            raise IOError(f"could not decode render from {url}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return gray, rgb, url, card
