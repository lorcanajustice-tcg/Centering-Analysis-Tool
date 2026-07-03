"""Disney Lorcana adapter."""
from __future__ import annotations

from .base import FrameLineSpec, GameSpec

LORCANA = GameSpec(
    name="lorcana",
    card_w_mm=63.5,
    card_h_mm=88.9,
    back_frame=FrameLineSpec(min_peak=45.0, search_mm=(0.5, 6.0),
                             nominal_border_mm=2.4),
    # Elsa 6/C2 back measured L+R total ~4.76mm; used only for the
    # borderless "equivalent ratio" convention. T+B total from four
    # standard-card backs measured 2026-07-03 (~4.6mm cut-to-frame).
    equiv_margin_lr_mm=4.76,
    equiv_margin_tb_mm=4.6,
    # Render-crop calibration (2026-07-03). Ravensburger renders
    # (1468x2048 via lorcanajson) are cropped inside trim NON-uniformly:
    # ~1.15mm total in width (symmetric: std-frame border is 59/59px, and
    # x cross-validated front-vs-back at the 0.01mm level), but ~2.3mm
    # total in height with MORE cropped at the TOP. Anchor survey over all
    # 3211 renders (footer emblem y=1899-1900 +-1px, frame top 66px,
    # footer band 1907px) shows the crop is layout-locked and constant
    # across sets/rarities incl. full-art (exceptions: Location cards and
    # Q1/Q2 promos have distinct layouts, same constancy). Vertical bias
    # calibrated against four standard-frame front+back pairs
    # (12:60, 12:91, 10:189, 12:191): +0.18 +- 0.06 (sem) mm, card
    # scatter +-0.13 (front-back registration + method residuals).
    # An Enchanted pair (6-8/C2) read higher (+0.4..0.7) but both its
    # photos carry a shadow-band edge artifact; single-constant model
    # retained pending cleaner captures. Uncertainty kept honest at 0.2.
    render_crop_bias_mm={"x": 0.0, "y": 0.20},
    render_crop_bias_unc_mm={"x": 0.05, "y": 0.20},
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
        m = re.fullmatch(r"(\d+)\s*/\s*(C?\d+\w*)", card_id.strip(),
                         re.IGNORECASE)
        if m and m.group(2).upper().startswith("C"):
            num, grp = int(m.group(1)), m.group(2).upper()
            hits = [c for c in cards if c.get("number") == num
                    and (c.get("promoGrouping") or "").upper() == grp]
            if len(hits) == 1:
                return hits[0]
            raise CardNotFound(f"{card_id}: {len(hits)} matches for promo lookup")
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
