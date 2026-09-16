"""Disney Lorcana adapter."""
from __future__ import annotations

from .base import FrameLineSpec, GameSpec, HexAnchorSpec, HexLayout

# Ink-cost hexagon layouts (2026-09-16). Survey of all 3,226 official
# pictures: calibration/hex_anchor_survey.py -> card_db/hex_anchors_report.json.
# Inside a layout the centre is fixed to ~0.005px (robust sd). Ring-edge
# apothems split into vertical and slanted sides from a 401-picture
# re-fit: the printed hexagons are not exactly regular.
# render_px_per_mm: the official picture's own print scale, 23.64 +-0.03
# px/mm from the render-to-cut widths of eight clean fronts (2026-07-06
# reshoot + fixtures; x totals 0.81-0.90mm). The picture is square-pixelled:
# the same scale predicts the hexagon's top-edge distance on those fronts
# to 0.005mm and its printed size to 0.16% (the known front-rim excess).
# The top/bottom render-to-cut totals would suggest 23.72, but they carry
# the picture's looser vertical crop, and 23.72 misses the hexagon's
# top-edge distance by 0.12mm.
LORCANA_HEX = HexAnchorSpec(
    render_size=(1468, 2048),
    render_px_per_mm=23.64,
    render_px_per_mm_rel_unc=0.001,
    layouts={
        "inkable": HexLayout(
            "inkable", (146.560, 156.788),
            ((-0.007, +0.0105, 57.522, 57.849),
             (+0.007, -0.0105, 66.446, 66.625)), swirl=True),
        "uninkable": HexLayout(
            "uninkable", (146.302, 155.565),
            ((-0.1285, -0.010, 67.863, 67.614),
             (+0.1285, +0.010, 78.555, 78.548))),
        # the six set-10 parchment-frame uninkable cards: same centre,
        # hexagon 11.7% bigger. Only used when the card is known.
        "uninkable_large": HexLayout(
            "uninkable_large", (146.334, 155.545),
            ((-0.100, 0.0, 75.726, 75.632),
             (+0.100, 0.0, 87.521, 87.749))),
    },
    default_layouts=("inkable", "uninkable"),
    # sets 1-3 printed the uninkable hexagon 3.3px (0.14mm) further right
    # (frame and footer emblem unchanged, so a layout change, not crop)
    older_layouts={("uninkable", ("1", "2", "3")): (149.598, 155.567)},
    percard_csv="hex_anchors_percard.csv",
)

LORCANA = GameSpec(
    name="lorcana",
    # MANUFACTURED size, not the 2.5x3.5" nominal (corrected 2026-08-21).
    # TAG's public DIG reports publish their own measured dimensions:
    # T6453597 2.475x3.460", Y3106454 2.476x3.462", M5576586 2.480x3.456"
    # = 62.865/62.890/62.992 x 87.884/87.935/87.782 mm (mean 62.92 x 87.87,
    # per-card spread ~0.13mm, i.e. bigger than this constant's rounding).
    # Independently corroborated: our chromaticity edges on two of those
    # scans give 62.864/62.884 x 87.894/87.929mm at a scan scale of exactly
    # 68.000 px/mm, agreeing with TAG to <=0.010mm on all four dimensions.
    # The old 63.5 x 88.9 was the 2.5x3.5" nominal, 1.0%/1.1% too big, and
    # made every absolute mm the tool printed 1% high. Ratios are
    # scale-free, so no centering percentage moved.
    # Aspect consequence: nominal H/W is 1.3975, not 1.4000 (the aspect
    # gates in back.py/borderless.py read it from these two constants).
    card_w_mm=62.9,
    card_h_mm=87.9,
    back_frame=FrameLineSpec(min_peak=45.0, search_mm=(0.5, 6.0),
                             nominal_border_mm=2.4),
    # Equivalence-margin convention totals (cut-to-frame, both sides),
    # re-derived 2026-07-06 from FIVE white-background backs (12:147,
    # 12:24, 12:133, 12:96, 12-54-P3): L+R 4.775 +- 0.017 (sd) mm --
    # remarkably manufacture-constant -- and T+B 4.305 +- 0.078 (sd) mm.
    # (The former T+B figure of 4.6 came from the 2026-07-03 dark-mat
    # shoot, whose horizontal edges were shadow-inflated outward.)
    # RESCALED 2026-08-21 with the card-size correction above: those mm
    # came out of the pipeline under card_w/h = 63.5/88.9, so they carried
    # the same 1% error. x *= 62.9/63.5 -> 4.735, y *= 87.9/88.9 -> 4.262.
    # This is a UNIT CONVERSION of the same 2026-07-06 measurement, not a
    # re-measurement; the separate re-derivation with colour edges (they
    # were measured with brightness edges, which sit ~0.07mm/side outside
    # the cut) still needs a capture on a coloured surface. See TODO.md.
    equiv_margin_lr_mm=4.735,
    equiv_margin_tb_mm=4.262,
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
    # (Same 2026-08-21 rescaling applies in principle - y *= 87.9/88.9
    # gives -0.0791 +- 0.0989 - but both round to the values already
    # quoted, so they are left as they stand.)
    render_crop_bias_mm={"x": 0.0, "y": -0.08},
    render_crop_bias_unc_mm={"x": 0.05, "y": 0.10},
    # Render-span gate bounds. Empirical over the seven clean 2026-07-06
    # white/kraft captures (five reshoot fronts + Gadget white/kraft +
    # Ursula 3/D23): x totals 0.73-0.90mm, y totals 1.21-1.97mm, max
    # clean single side 1.33mm; margins ~0.3mm. Motivating failure: a
    # hard cast shadow hugging a dark full-art top edge fakes a sharp
    # "cut" (IMG_6416: top edge measured +2.24mm outside the render,
    # y-total 2.77mm) that per-line edge QA cannot distinguish locally.
    # These are padded round numbers around the observed range, and the
    # 2026-08-21 rescaling moves them by <=0.02mm (x_total 0.495..1.139,
    # y_total 1.038..2.126) - inside the padding and inside the rounding
    # they are quoted at, so they are left as they stand. Both the bounds
    # and the spans they gate shrink by the same 1%, so no verdict moves.
    render_span_bounds_mm={"x_total": (0.50, 1.15),
                           "y_total": (1.05, 2.15),
                           "side": (-0.10, 1.90)},
    # Face-aware cut definition (2026-08-21). Same physical card, same
    # scanner: the two TAG backs measure 62.864/62.884 x 87.894/87.929mm
    # and agree with TAG's own dimensions to <=0.010mm, while the FRONTS
    # of those same two cards measure 0.100/0.107mm wider and 0.180/0.150mm
    # taller. The card did not change - the full-art cut-edge rim did
    # (dark on one edge, bright on the other). Half of the excess per side:
    # 0.05mm in x, 0.08mm in y, carried as UNCERTAINTY, not subtracted as a
    # correction - two cards on one scanner do not calibrate a bias, and
    # the rim's sign is not the same on opposite edges.
    # The back's own term is left at zero: the DIG comparison bounds it
    # below 0.010mm, under the detector term already in edge_def_px.
    cut_def_mm={"front": {"x": 0.05, "y": 0.08}},
    hex_anchor=LORCANA_HEX,
)

ALLCARDS_URL = "https://lorcanajson.org/files/current/en/allCards.json"


import re
from typing import Optional

import cv2
import numpy as np

from ..cache import DiskCache

ALLCARDS_ZIP_URL = "https://lorcanajson.org/files/current/en/allCards.json.zip"


import json
from pathlib import Path as _Path


class CardNotFound(ValueError):
    """Card-id lookup failure.

    n_matches is 0 for "not found", >1 for ambiguous ids, and None when
    the card database itself was unavailable (e.g. offline first run).
    """

    def __init__(self, msg, n_matches=None):
        super().__init__(msg)
        self.n_matches = n_matches


def _variant_from_full_id(rec: dict):
    """Variant letter from a fullIdentifier token (e.g. "24A/P2 ..." -> "A").

    card_db/index.json records carry no explicit "variant" field; when a
    card has one, the letter is embedded in the identifier token.
    """
    number = str(rec.get("number"))
    token = (rec.get("fullIdentifier") or "").split("\u2022")[0].split("/")[0].strip()
    if token and token != number and token.startswith(number):
        return token[len(number):] or None
    return None


def _default_local_db_dir():
    """The repo's card_db/ (3,211+ local official renders), when present."""
    d = _Path(__file__).resolve().parents[3] / "card_db"
    return d if (d / "index.json").exists() else None


class LorcanaRenderSource:
    """Official Ravensburger renders located via lorcanajson.org.

    lorcanajson includes promos that other APIs miss. Accepted card ids:
    - "6/C2"           number / promo grouping
    - "C2-6", "8-210"  unified SET-NUMBER form
    - "7:69"           setCode : number
    - "Elsa - Ice Maker" (name or "name - version" substring, unique match)

    Lookup is layered so a failed lookup never blocks an analysis that
    could still run (2026-07-08, after a stale cache refused 10/C2):
    1. cached allCards.json (7-day TTL);
    2. on a ZERO-match only, one forced re-download -- promos are added
       upstream continually, so a miss usually means the cache predates
       the card (ambiguous ids are real and are not retried);
    3. the local card_db index + images (same source, works offline).
    """

    def __init__(self, cache=None, local_db_dir=None):
        self.cache = cache or DiskCache()
        self._cards = None
        self._local_cards = None
        self._refreshed = False
        self.local_db_dir = (_Path(local_db_dir) if local_db_dir
                             else _default_local_db_dir())

    def _load(self):
        if self._cards is None:
            j = self.cache.fetch_json_maybe_zipped(ALLCARDS_ZIP_URL)
            self._cards = j["cards"]
        return self._cards

    def _load_local(self):
        d = getattr(self, "local_db_dir", None)
        if not d:
            return None
        cached = getattr(self, "_local_cards", None)
        if cached is not None:
            return cached
        idx = _Path(d) / "index.json"
        if not idx.exists():
            return None
        recs = json.loads(idx.read_text(encoding="utf-8")).get("cards") or []
        for r in recs:
            if "variant" not in r:
                r["variant"] = _variant_from_full_id(r)
            f = _Path(d) / (r.get("file") or "")
            r["_local_file"] = str(f) if r.get("file") and f.exists() else None
        self._local_cards = recs
        return recs

    def _match(self, cards, card_id: str) -> dict:
        """Resolve card_id within a given card list (raises CardNotFound)."""
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
                f"\"{card_id}\" matches {len(hits)} promo cards, so it is "
                "not clear which one you mean"
                + (f". Did you mean one of these? "
                   f"{[c.get('fullIdentifier') for c in hits[:5]]}"
                   if hits else ". Check the set and number."),
                n_matches=len(hits))
        m = re.fullmatch(r"(\w+)\s*:\s*(\d+)", card_id.strip())
        if m:
            sc, num = m.group(1), int(m.group(2))
            hits = [c for c in cards if str(c.get("setCode")) == sc
                    and c.get("number") == num and not c.get("promoGrouping")]
            if len(hits) == 1:
                return hits[0]
            raise CardNotFound(
                f"No single card matches \"{card_id}\" ({len(hits)} found). "
                "Check the set and the card number.", n_matches=len(hits))
        # Unified "SET-NUMBER" form (hyphen): the set identifier is either a
        # set code (1-13, Q1, Q2) or a promo grouping (C2, P1, D23, ...).
        # Those two namespaces are disjoint, so the token is unambiguous:
        # "8-210" -> set 8 / card 210; "C2-6", "P1-42" -> promo grouping.
        m = re.fullmatch(r"([A-Za-z0-9]+)\s*-\s*(\d+)", card_id.strip())
        if m:
            pre, num = m.group(1), int(m.group(2))
            hits = [c for c in cards if str(c.get("setCode")) == pre
                    and c.get("number") == num and not c.get("promoGrouping")]
            if len(hits) == 1:
                return hits[0]
            ph = [c for c in cards
                  if (c.get("promoGrouping") or "").upper() == pre.upper()
                  and c.get("number") == num]
            if len(ph) == 1:
                return ph[0]
            total = len(hits) + len(ph)
            raise CardNotFound(
                f"No single card matches \"{card_id}\" ({total} found). "
                "Check the set and the card number"
                + (f". Did you mean one of these? "
                   f"{[c.get('fullIdentifier') for c in ph[:5]]}"
                   if len(ph) > 1 else "."), n_matches=total)
        q = card_id.strip().lower()
        hits = [c for c in cards
                if q in f"{c.get('name','')} - {c.get('version','')}".lower()]
        if len(hits) == 1:
            return hits[0]
        raise CardNotFound(
            (f"\"{card_id}\" matches {len(hits)} cards by name, so it is "
             "not clear which one you mean. Did you mean one of these? "
             f"{[c.get('fullIdentifier') for c in hits[:5]]}"
             if hits else
             f"No card matches \"{card_id}\". Type the set and number, "
             "like 8-210, or more of the card's name."),
            n_matches=len(hits))

    def resolve(self, card_id: str) -> dict:
        # 1) cached allCards.json
        try:
            cards = self._load()
        except Exception as e:
            err = CardNotFound(f"{card_id}: card database unavailable ({e})")
        else:
            try:
                return self._match(cards, card_id)
            except CardNotFound as e:
                err = e
        # 2) zero matches may just mean a stale cache: force one refresh.
        cache = getattr(self, "cache", None)
        if (cache is not None and not getattr(self, "_refreshed", False)
                and err.n_matches in (0, None)):
            try:
                self._cards = cache.fetch_json_maybe_zipped(
                    ALLCARDS_ZIP_URL, ttl_days=0)["cards"]
                self._refreshed = True
                return self._match(self._cards, card_id)
            except CardNotFound as e:
                err = e
            except Exception:
                pass  # offline: fall through to the local card_db
        # 3) local card_db (same upstream source; works offline).
        if err.n_matches in (0, None):
            local = self._load_local()
            if local:
                try:
                    return self._match(local, card_id)
                except CardNotFound:
                    pass
        raise err

    def _local_file_for(self, card: dict):
        """Path of the local card_db render for this card, if we have it."""
        if card.get("_local_file"):
            return card["_local_file"]
        local = self._load_local()
        if not local:
            return None
        key = (card.get("number"), (card.get("promoGrouping") or "").upper(),
               str(card.get("setCode")), (card.get("variant") or "").upper())
        for r in local:
            if (r.get("number"), (r.get("promoGrouping") or "").upper(),
                    str(r.get("setCode")),
                    (r.get("variant") or "").upper()) == key:
                return r.get("_local_file")
        return None

    def get_render(self, card_id: str):
        """Returns (render_gray float32, render_rgb uint8, url, card_dict).

        Prefers the local card_db image: it was downloaded from the same
        images.full URL (byte-identical official render, and the one the
        detector SIFT-verified against), and it works offline. Falls back
        to fetching the official URL via the DiskCache.
        """
        card = self.resolve(card_id)
        lf = self._local_file_for(card)
        if lf:
            bgr = cv2.imread(lf, cv2.IMREAD_COLOR)
            if bgr is not None:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
                return gray, rgb, lf, card
        url = (card.get("images") or {}).get("full") or card.get("url")
        if not url:
            raise CardNotFound(
                f"There is no official picture available for {card_id}, so "
                "the front cannot be measured. The back can still be "
                "checked on its own.")
        p = self.cache.fetch(url, suffix=".img")
        buf = np.frombuffer(p.read_bytes(), np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            raise IOError("The official picture of this card downloaded, but "
                      "could not be opened. Try again in a moment.")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return gray, rgb, url, card
