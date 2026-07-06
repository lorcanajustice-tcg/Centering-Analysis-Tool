"""Combined card report: both faces + front-back registration cross-check."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

from .back import analyze_back
from .borderless import analyze_borderless
from .games.base import GameSpec
from .types import CardResult, Measurement, QAFlag, Uncertainty


def analyze_card(back_photo: Optional[str] = None,
                 front_photo: Optional[str] = None,
                 card_id: Optional[str] = None,
                 game: GameSpec = None,
                 render_source=None,
                 out_dir: Optional[str] = None) -> CardResult:
    if back_photo is None and front_photo is None:
        raise ValueError("need at least one photo")
    if front_photo is not None and card_id is None:
        raise ValueError("borderless front analysis requires card_id")

    res = CardResult(game=game.name)
    if back_photo:
        res.back = analyze_back(back_photo, game, out_dir=out_dir)
    if front_photo:
        res.front = analyze_borderless(front_photo, card_id, game,
                                       render_source=render_source,
                                       out_dir=out_dir)

    if res.back and res.front:
        res.registration_mm = {}
        # X axis: the same physical cut seen from both faces.
        # Front: shift_x < 0 means print displaced toward the LEFT card edge.
        # Back (mirrored horizontally): frame displaced toward the back-RIGHT
        # edge by (L-R)/2, which is the front-LEFT direction.
        bl = res.back.borders_mm.get("left")
        br = res.back.borders_mm.get("right")
        fx = res.front.shift_mm.get("x")
        if (bl and br and fx and bl.status == br.status == fx.status == "measured"):
            front_toward_left = -fx.value
            back_toward_front_left = (bl.value - br.value) / 2.0
            delta = front_toward_left - back_toward_front_left
            sig = Uncertainty(
                statistical=math.sqrt(
                    fx.uncertainty.statistical ** 2
                    + (bl.uncertainty.statistical ** 2
                       + br.uncertainty.statistical ** 2) / 4.0),
                perspective=math.sqrt(
                    fx.uncertainty.perspective ** 2
                    + (bl.uncertainty.perspective ** 2
                       + br.uncertainty.perspective ** 2) / 4.0),
                edge_definition=math.sqrt(
                    fx.uncertainty.edge_definition ** 2
                    + (bl.uncertainty.edge_definition ** 2
                       + br.uncertainty.edge_definition ** 2) / 4.0))
            res.registration_mm["x"] = Measurement(delta, "mm", sig)
            if abs(delta) > 2.0 * sig.total:
                res.qa.append(QAFlag(
                    "REGISTRATION_DISCREPANCY",
                    f"front and back disagree on the horizontal cut position "
                    f"by {delta:+.3f}mm (> 2 sigma); check for sleeve, curl, "
                    "or a mis-detected edge on either face"))
            fpct = res.front.equivalent_ratio_lr
            bpct = res.back.ratio_lr
            if fpct and bpct and fpct.first_pct and bpct.first_pct:
                res.mirror_consistency = (
                    f"back L/R {bpct.display} vs front equivalent "
                    f"{fpct.display} (mirrored {100-fpct.first_pct:.1f}/"
                    f"{fpct.first_pct:.1f}): one cut, "
                    f"registration {delta:+.3f}mm")
        else:
            res.registration_mm["x"] = Measurement.refused(
                "mm", "requires measured back L/R borders and front x-shift")
        by_t = res.back.borders_mm.get("top")
        by_b = res.back.borders_mm.get("bottom")
        fy = res.front.shift_mm.get("y")
        if (by_t and by_b and fy
                and by_t.status == by_b.status == fy.status == "measured"):
            front_toward_top = -fy.value
            # T/B does not mirror under the vertical-axis flip. The back
            # frame is displaced TOWARD THE TOP when the top border is the
            # smaller one: (B-T)/2 (sign fixed 2026-07-06; (T-B)/2 is the
            # +y-down frame offset, i.e. toward the bottom).
            back_toward_top = (by_b.value - by_t.value) / 2.0
            delta = front_toward_top - back_toward_top
            sig = Uncertainty(
                statistical=math.sqrt(
                    fy.uncertainty.statistical ** 2
                    + (by_t.uncertainty.statistical ** 2
                       + by_b.uncertainty.statistical ** 2) / 4.0),
                perspective=math.sqrt(
                    fy.uncertainty.perspective ** 2
                    + (by_t.uncertainty.perspective ** 2
                       + by_b.uncertainty.perspective ** 2) / 4.0),
                edge_definition=math.sqrt(
                    fy.uncertainty.edge_definition ** 2
                    + (by_t.uncertainty.edge_definition ** 2
                       + by_b.uncertainty.edge_definition ** 2) / 4.0))
            res.registration_mm["y"] = Measurement(delta, "mm", sig)
            if abs(delta) > 2.0 * sig.total:
                res.qa.append(QAFlag(
                    "REGISTRATION_DISCREPANCY",
                    f"front and back disagree on the vertical cut position "
                    f"by {delta:+.3f}mm (> 2 sigma); genuine front-back "
                    "print registration reaches this level (scatter "
                    "+-0.19mm), but check for curl or a mis-detected edge"))
        else:
            reasons = []
            for nm, m in (("back top", by_t), ("back bottom", by_b),
                          ("front y", fy)):
                if m is None or m.status != "measured":
                    reasons.append(f"{nm} unavailable"
                                   + (f" ({m.refusal_reason})" if m and m.refusal_reason else ""))
            res.registration_mm["y"] = Measurement.refused(
                "mm", "; ".join(reasons))
    return res
