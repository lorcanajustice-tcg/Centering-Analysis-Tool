"""Borderless-face pipeline: measure the print offset of the physical cut by
matching the photo to the official render.

Key caveats (documented in the result):
- official renders are cropped inside true trim (~0.6mm/side horizontally,
  more vertically), so per-side offsets vs render bounds are NOT absolute
  borders.
- the vertical crop is ASYMMETRIC (more cropped at the top). The calibrated
  per-axis bias in GameSpec.render_crop_bias_mm is subtracted from the raw
  shift; its systematic uncertainty propagates into the result. The crop is
  layout-locked across renders (anchor survey over all 3211 Lorcana
  renders), so one constant per game/axis applies.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np

from . import edges as E
from . import geometry as G
from .fitting import (_colour_edge, _edge_report, _frame_proximity_qa,
                      _prefer_fit, _shadow_band_qa, _thin_fit)
from .estimate import BASE_SYSTEMATIC_MM, CAP_MM, rescue_edge
from .hexanchor import equivalent_ratio, measure_in_card
from .games.base import GameSpec
from .imgio import digital_background, digital_image_reason, load_photo
from .infer import OPPOSITE, infer_missing_edge
from .locate import background_uniformity, card_component_bbox, coarse_locate
from .overlay import C_EDGE, C_FRAME, Overlay
from .types import (BorderlessResult, EdgeFitReport, Measurement, QAFlag,
                    Ratio, RenderMatchReport, TiltReport, Uncertainty)
from .render_match import match_to_render
from .uncertainty import compose_ratio_uncertainty

# fourth-edge handling (see infer.py)
# hybrid cut detector as a last resort, anchored on a construction
CUT_RESCUE_WINDOWS_MM = (1.2, 2.0)
CUT_RESCUE_RMS_PX = 1.5
CUT_RESCUE_AGREE_MM = 0.10
RESCAN_MIN_MM = 1.0      # smallest half-window for the second look
RESCAN_K = 3.0           # second-look window / agreement, in sigmas
# ...but never demand agreement tighter than this: the edge found on the
# second look is a full strict-tier fit, and still has to pass the
# leaning-edge check and the render-span gate afterwards. Belle IMG_2033's
# real bottom lands 1.37mm from the pair-scaled prediction (the photo is a
# 9:16 re-encode; its quad is ~1% taller than a card).
RESCAN_AGREE_MIN_MM = 1.5
INFER_MAX_SEED_DEV_MM = 10.0  # constructed edge vs where the card was found

# leaning-edge check (official-picture space). Good fixture fronts: every
# edge within 0.29 deg of the shared rotation (2026-09-15, six fronts).
# Belle IMG_2033's right edge (a case edge): 1.56 deg.
LEAN_MAX_DEG = 1.0
LEAN_MIN_INLIERS = 80  # below this the alignment is too loose to judge


def _lean_check(Hpr, lines, methods):
    """Measured edges whose rotation in the official picture's frame
    differs from the shared (median) rotation by more than LEAN_MAX_DEG,
    as {side: plain-English reason}."""
    rot = {}
    for side, line in lines.items():
        if line is None or methods.get(side) == "inferred":
            continue
        pts = G.transform_points(Hpr, line.points(60))
        if side in ("left", "right"):
            m = np.polyfit(pts[:, 1], pts[:, 0], 1)[0]
            rot[side] = -math.degrees(math.atan(m))
        else:
            m = np.polyfit(pts[:, 0], pts[:, 1], 1)[0]
            rot[side] = math.degrees(math.atan(m))
    if len(rot) < 3:
        return {}
    ref = float(np.median(list(rot.values())))
    out = {}
    for side, r in rot.items():
        if abs(r - ref) > LEAN_MAX_DEG:
            out[side] = (
                f"leans {abs(r - ref):.1f} degrees away from the other "
                "edges once the photo's angle is taken out, and a real "
                "card's edges are square to each other, so it is not the "
                "edge of the card")
    return out


_SIDES = ("left", "right", "top", "bottom")

# The advice given whenever a photo simply is not good enough to measure.
RESHOOT = ("Re-shoot the card flat and unsleeved on plain white paper, with "
           "soft light coming from all sides, and try again.")

# NOTE (2026-07-03): the render crop is NOT symmetric about the print
# centre vertically. Per-axis bias and its systematic uncertainty now come
# from GameSpec.render_crop_bias_mm / render_crop_bias_unc_mm (calibrated;
# see games/lorcana.py). The historical 0.03mm allowance applied to both
# axes only ever validated the x axis.


def _render_span_violations(offsets_mm: dict, bounds: dict,
                             slack_mm: Optional[dict] = None) -> dict:
    """Physical-plausibility check of the fitted quad against the render.

    The cut always lies OUTSIDE the render bounds (the render is cropped
    inside trim, crop >= 0) and the left+right / top+bottom render-to-cut
    totals are layout-locked render-crop constants plus the manufacture
    cut size - independent of where the cut landed on this card. A fitted
    edge dragged off the cut by a hard cast shadow or foil curl breaks
    these bounds even when its fit statistics look clean (a cast shadow's
    outer boundary is as sharp as a real cut and tonally continuous with
    dark card art). Returns {axis: reason}; empty when plausible.

    `slack_mm` widens the bounds for edges that were only estimated or
    worked out ({side: mm}, typically 2 sigma of the edge's own extra
    uncertainty): the gate asks whether the edge COULD be the cut, and an
    estimate is only claimed to within its margin.
    """
    out = {}
    slack_mm = slack_mm or {}
    for ax, (a, b) in (("x", ("left", "right")), ("y", ("top", "bottom"))):
        oa = offsets_mm[f"{a}_outside_render"]
        ob = offsets_mm[f"{b}_outside_render"]
        t_lo, t_hi = bounds[f"{ax}_total"]
        tot_slack = slack_mm.get(a, 0.0) + slack_mm.get(b, 0.0)
        t_lo, t_hi = t_lo - tot_slack, t_hi + tot_slack
        probs = []
        for n, v in ((a, oa), (b, ob)):
            s_lo = bounds["side"][0] - slack_mm.get(n, 0.0)
            s_hi = bounds["side"][1] + slack_mm.get(n, 0.0)
            if not s_lo <= v <= s_hi:
                probs.append(f"the {n} edge sits {v:+.2f}mm outside the "
                             f"official picture, where every real card "
                             f"falls between {s_lo:+.2f} and {s_hi:+.2f}mm")
        if not t_lo <= oa + ob <= t_hi:
            probs.append(f"the {a} and {b} edges together sit "
                         f"{oa + ob:.2f}mm outside the official picture, "
                         f"where every real card falls between {t_lo:.2f} "
                         f"and {t_hi:.2f}mm")
        if probs:
            out[ax] = "; ".join(probs)
    return out


DIGITAL_ADVICE = ("Photograph the printed card itself: unsleeved, flat on "
                  "plain white paper, with soft light from all sides.")


def analyze_borderless(photo, card_id, game, *args, **kw) -> BorderlessResult:
    """Front of a card: see _analyze_borderless. A digital picture (not a
    photo) that cannot be measured gets that said plainly, instead of the
    edge finder's guess at what went wrong."""
    res = _analyze_borderless(photo, card_id, game, *args, **kw)
    return explain_digital_refusals(res)


def explain_digital_refusals(res):
    """Swap the refusal reasons on a digital picture for the real one."""
    digital = next((q for q in res.qa if q.code == "DIGITAL_IMAGE"), None)
    if digital is None:
        return res
    for store in (res.shift_mm,):
        for k, m in store.items():
            if m.status == "refused":
                store[k] = Measurement.refused(
                    m.unit, digital.message + " (The edge finder said: "
                    + (m.refusal_reason or "no edge") + ")", DIGITAL_ADVICE)
    for name in ("equivalent_ratio_lr", "equivalent_ratio_tb"):
        r = getattr(res, name)
        if r is not None and r.status == "refused":
            setattr(res, name, Ratio.refused(r.axis, digital.message,
                                             DIGITAL_ADVICE))
    return res


def _analyze_borderless(photo: str | Path, card_id: Optional[str],
                        game: GameSpec,
                       render_source=None, out_dir: Optional[str] = None,
                       n_scans: int = 50, make_overlay: bool = True,
                       manual_bbox: Optional[tuple] = None
                       ) -> BorderlessResult:
    rgb, gray, inp = load_photo(photo)
    Himg, Wimg = gray.shape
    res = BorderlessResult(kind="borderless", game=game.name, input=inp,
                           tilt=TiltReport())
    qa = res.qa
    digital = digital_background(rgb)
    if digital:
        qa.append(QAFlag("DIGITAL_IMAGE", digital_image_reason(digital)))

    if render_source is None and card_id:
        from .games.lorcana import LorcanaRenderSource
        render_source = LorcanaRenderSource()

    # --- coarse localization: the card is the odd-one-out component
    # (bright on a dark mat, or dark/bordered on a light background); when
    # segmentation is ambiguous (e.g. mid-tone artwork on a similar-toned
    # background) fall back to per-side coarse scans ---
    seed_slop_mm = 2.5
    if manual_bbox is not None:
        # human-supplied coarse localization (drag-box). Humans are
        # excellent at the segmentation step that fails on low-contrast/
        # textured backgrounds; everything downstream (sub-pixel fits,
        # tilt correction, gates, estimate tier) runs unchanged. The
        # box is approximate, so the fine search window is widened.
        x0, y0, x1, y1 = (int(round(float(v))) for v in manual_bbox)
        if not (0 <= x0 < x1 <= Wimg and 0 <= y0 < y1 <= Himg):
            raise ValueError("The box you drew falls outside the photo. "
                             "Draw it again, inside the picture.")
        seed_slop_mm = 3.5
        qa.append(QAFlag(
            "MANUAL_LOCALIZATION",
            "You drew the box that says where the card is. That only tells "
            "the analyser where to look - the measuring, the checks and "
            "the accuracy are all exactly the same as normal.",
            severity="info"))
    else:
        try:
            x0, y0, x1, y1 = card_component_bbox(gray)
        except RuntimeError as err:
            coarse, _ = coarse_locate(gray, game.card_w_mm, game.card_h_mm)
            bad = [s for s in _SIDES if coarse[s].pos is None]
            if bad:
                raise RuntimeError(
                    (digital_image_reason(digital) + " " if digital else "")
                    + "The card could not be found in this photo. "
                    + " ".join(f"On the {s} edge, {coarse[s].reason}."
                               for s in bad)
                    + " Try again on plain white paper with soft, even "
                    "light - or use \"Mark the card position\" below to "
                    "draw a box around the card yourself."
                    ) from err
            x0, y0 = int(coarse["left"].pos), int(coarse["top"].pos)
            x1, y1 = int(coarse["right"].pos), int(coarse["bottom"].pos)
            slops = [coarse[s].slop_mm for s in _SIDES]
            if all(v is not None for v in slops):
                # a seed that knows its own accuracy (tight-crop path):
                # a narrow window keeps the level estimates off distant
                # artwork, which on a full-bleed face is what breaks them
                seed_slop_mm = min(slops)
    ppm0 = (x1 - x0) / game.card_w_mm

    chroma = E.chromaticity(rgb) if rgb is not None and rgb.ndim == 3 else None

    corners_bg = background_uniformity(gray)
    vals = list(corners_bg.values())
    if max(vals) > 1.8 * max(min(vals), 1e-6):
        qa.append(QAFlag("BACKGROUND_NONUNIFORM",
                         "The background is much brighter in some corners "
                         f"than others (from {min(vals):.0f} to "
                         f"{max(vals):.0f} on a 0-255 brightness scale). "
                         "The parts of each edge that fall in the brightest "
                         "or darkest areas were skipped."))

    # --- physical edges: sub-pixel brightness step ---
    rows = np.linspace(y0 + 0.15 * (y1 - y0), y0 + 0.85 * (y1 - y0), n_scans)
    cols = np.linspace(x0 + 0.15 * (x1 - x0), x0 + 0.85 * (x1 - x0), n_scans)
    approx = {"left": x0, "right": x1, "top": y0, "bottom": y1}
    lines, reports, methods, est_extra = {}, {}, {}, {}

    def _measure_side(side, approx_v, s_out, s_in, qa):
        """Fit one cut edge: strict tier (step, texture, colour), then the
        estimate tier. Returns (line|None, report, diag, method, est_unc)."""
        us = rows if side in ("left", "right") else cols
        est_unc = None
        u_ok, v_ok, diag = E.step_scan(
            gray, side, approx_v, us,
            search_out_px=s_out,
            search_in_px=s_in)
        line, rep = _edge_report(side, "step", u_ok, v_ok, diag)
        method = "step"
        u2, v2, d2 = np.array([]), np.array([]), None
        if _thin_fit(line, diag):
            # a black border on a dark textured mat has no brightness step
            # at all, and a full-bleed face whose artwork happens to match
            # the surround leaves the step scanner clinging to a handful of
            # lines. The texture transition marks the same cut; keep
            # whichever fit the edge actually supports.
            u2, v2, d2 = E.texture_scan(
                gray, side, approx_v, us,
                search_out_px=s_out,
                search_in_px=s_in)
            line2, rep2 = _edge_report(side, "texture", u2, v2, d2)
            line, rep, diag, method = _prefer_fit(
                (line, rep, diag, "step"), (line2, rep2, d2, "texture"))
        # chromaticity edge: on a coloured background the shadow at the cut
        # is a brightness ramp but not a hue change, and a full-bleed face
        # can carry a dark cut-edge rim on one side and a bright one on the
        # other - both displace a brightness scan, asymmetrically.
        line, rep, diag, method = _colour_edge(
            chroma, side, approx_v, us,
            s_out, s_in,
            (line, rep, diag, method), qa, ppm0)
        if line is None:
            # estimate tier: the strict tier refused this edge; attempt an
            # explicitly-labelled rescue (cluster split + hybrid cut
            # cross-check arbitration). Never silent: the report keeps the
            # strict refusal, the edge fit is status="estimated", and the
            # axis result is capped at CAP_MM total uncertainty.
            strict_notes = "; ".join(rep.notes) or "refused"
            tries = [("step", u_ok, v_ok), ("texture", u2, v2)]
            for mname, uu, vv in tries:
                if len(uu) < 4:
                    continue
                est, why = rescue_edge(gray, side, uu, vv, us, ppm0)
                if est is None:
                    note = f"tried to estimate it instead, but {why}"
                    if note not in rep.notes:
                        rep.notes.append(note)
                    continue
                line = est.line
                rep = EdgeFitReport(
                    edge=side, method=f"{mname}-estimate",
                    n_points=line.n, n_rejected=len(uu) - line.n,
                    rms_residual_px=line.rms,
                    angle_deg=line.angle_from_nominal_deg(),
                    bow_px=line.bow_px, status="estimated",
                    notes=[f"could not be measured properly: "
                           f"{strict_notes}", est.note])
                method = mname
                est_unc = est.extra_unc_mm
                qa.append(QAFlag(
                    "EDGE_ESTIMATED",
                    f"The {side} edge was too unclear to measure properly, "
                    f"so it has been estimated instead: {est.note}. The "
                    "result is marked with a squiggle and carries a bigger "
                    "margin of error to match."))
                break
        if line is not None and line.bow_px and line.bow_px > 3.0:
            qa.append(QAFlag("CURL_SUSPECTED",
                             f"The {side} edge is not straight - it curves "
                             f"by {line.bow_px:.1f} pixels along its "
                             "length. A bent card makes its edge measure in "
                             "the wrong place."))
        if diag.n_ok and diag.n_ok < 0.7 * diag.n_attempted:
            qa.append(QAFlag("EDGE_PARTIALLY_EXCLUDED",
                             f"On the {side} edge, "
                             f"{diag.n_attempted - diag.n_ok} of "
                             f"{diag.n_attempted} readings had to be thrown "
                             f"out ({diag.summary()}). The measurement uses "
                             "the clean stretches only."))
        return line, rep, diag, method, est_unc

    for side in _SIDES:
        # the scanners need a minimum number of samples whatever the
        # resolution: at ~7 px/mm a 2.5mm window is too short to read
        win0 = max(seed_slop_mm * ppm0, E.MIN_SCAN_HALF_WINDOW_PX)
        line, rep, _, methods[side], est_u = _measure_side(
            side, approx[side], win0, win0, qa)
        if est_u is not None:
            est_extra[side] = est_u
        lines[side], reports[side] = line, rep
        res.edge_fits.append(rep)

    def _fourth_edge(side, force_rescan=False, no_rescan=False):
        """The opposite edge and the card's known size place a missing one.

        Without it the whole front refuses, even the axis whose two edges
        were measured cleanly. First the missing edge is re-scanned where
        the construction says it is (a sloppy seed - a hand-drawn box, a
        case edge - is the usual reason it was missed); only if that also
        fails is the constructed line itself used, as an estimate."""
        opp_side = OPPOSITE[side]
        p1, p2 = (("top", "bottom") if side in ("left", "right")
                  else ("left", "right"))
        ax_p = "y" if side in ("left", "right") else "x"
        ax_o = "x" if side in ("left", "right") else "y"

        def _ed(s, ax):
            return math.hypot(
                game.edge_def_mm("front", methods.get(s, "step"), ax, ppm0),
                est_extra.get(s, 0.0))

        # constructed edges are never inputs to another construction
        usable = {k: (v if methods.get(k) != "inferred" else None)
                  for k, v in lines.items()}
        opp = usable[opp_side]
        if opp is None:
            reports[side].notes.append(
                "the edge opposite it was not measured either, so its "
                "position could not be worked out")
            return
        kw = {}
        if n_inl >= LEAN_MIN_INLIERS:
            kw = {"H_photo_to_render": Hpr,
                  "h_unc_mm": med_err / math.sqrt(max(n_inl, 1)) * 3.0
                  / ppm_r}
            bounds = getattr(game, "render_span_bounds_mm", None)
            if bounds:
                # the picture's crop inside the cut is layout-locked, so
                # its own size gives a scale: good to ~0.5%
                ax = "x" if side in ("left", "right") else "y"
                t_lo, t_hi = bounds[f"{ax}_total"]
                dim = game.card_w_mm if ax == "x" else game.card_h_mm
                pic_mm = dim - 0.5 * (t_lo + t_hi)
                kw["frame_ppm"] = (Wr if ax == "x" else Hr) / pic_mm
                kw["frame_ppm_rel_unc"] = 0.5 * (t_hi - t_lo) / pic_mm
        pair_ok = usable[p1] is not None and usable[p2] is not None
        inf, why = infer_missing_edge(
            side, usable, game.card_w_mm, game.card_h_mm, (Wimg, Himg),
            sep_unc_mm=(math.hypot(_ed(p1, ax_p), _ed(p2, ax_p))
                        if pair_ok else 0.0),
            opp_unc_mm=math.hypot(
                _ed(opp_side, ax_o),
                opp.rms / math.sqrt(max(opp.n, 1)) / ppm0),
            **kw)
        rep = reports[side]
        if inf is None:
            rep.notes.append(f"tried to work it out from the other "
                             f"edges, but {why}")
            return
        mid_u = 0.5 * sum(inf.line.u_range)
        v_inf = float(inf.line.v_at(mid_u))
        seed_dev_mm = abs(v_inf - approx[side]) / inf.ppm

        # 1) re-scan around the constructed position, when it is far
        # enough from the original seed that the first scan missed it
        win = 0.0
        if not no_rescan and (force_rescan
                              or seed_dev_mm > 0.5 * seed_slop_mm):
            tilt_px = abs(inf.line.m) * 0.5 * (inf.line.u_range[1]
                                               - inf.line.u_range[0])
            win = max(RESCAN_MIN_MM * inf.ppm,
                      RESCAN_K * inf.extra_unc_mm * inf.ppm + tilt_px,
                      E.MIN_SCAN_HALF_WINDOW_PX)
            lim = Wimg if side in ("left", "right") else Himg
            win = min(win, v_inf - 1.0, lim - 2.0 - v_inf)
        if win >= E.MIN_SCAN_HALF_WINDOW_PX:
            qa2 = []
            line2, rep2, _, meth2, est2 = _measure_side(
                side, v_inf, win, win, qa2)
            if line2 is not None and est2 is None and rep2.status == "ok":
                d_mm = abs(float(line2.v_at(mid_u)) - v_inf) / inf.ppm
                if d_mm <= max(RESCAN_K * inf.extra_unc_mm,
                               RESCAN_AGREE_MIN_MM):
                    rep2.notes.append(
                        f"found on a second look, {seed_dev_mm:.1f}mm from "
                        "where the card was first placed, by searching "
                        "where the other three edges said it must be "
                        f"(it landed {d_mm:.2f}mm from that prediction)")
                    qa.extend(qa2)
                    qa.append(QAFlag(
                        "EDGE_RELOCATED",
                        f"The {side} edge was not where the card was first "
                        f"placed - it was {seed_dev_mm:.1f}mm away. It was "
                        "found by looking where the other three edges and "
                        "the card's size said it had to be, and then "
                        "measured normally.", severity="info"))
                    res.edge_fits[res.edge_fits.index(rep)] = rep2
                    reports[side], lines[side] = rep2, line2
                    methods[side] = meth2
                    return
                rep.notes.append(
                    f"a second look found an edge {d_mm:.2f}mm from where "
                    "the other three edges said it must be, which is too "
                    "far to trust")
            else:
                rep.notes.append(
                    "a second look where the other three edges said it "
                    "must be did not find a clean edge either")

        # 2) the hybrid cut detector, anchored on the construction. On its
        # own (seed-anchored) it latches onto artwork with clean-looking
        # fits whose position depends on the window, so it is only trusted
        # here: close to a prediction, and giving the same answer from two
        # different windows.
        cut = _cut_rescue(side, inf)
        if cut is not None:
            line_c, extra_c, note_c = cut
            strict_notes = "; ".join(rep.notes) or "refused"
            new_rep = EdgeFitReport(
                edge=side, method="cut-estimate", n_points=line_c.n,
                n_rejected=line_c.n_rej, rms_residual_px=line_c.rms,
                angle_deg=line_c.angle_from_nominal_deg(),
                bow_px=line_c.bow_px, status="estimated",
                notes=[f"could not be measured properly: {strict_notes}",
                       note_c])
            res.edge_fits[res.edge_fits.index(rep)] = new_rep
            reports[side], lines[side] = new_rep, line_c
            methods[side] = "cut"
            est_extra[side] = extra_c
            qa.append(QAFlag(
                "EDGE_ESTIMATED",
                f"The {side} edge was too unclear to measure properly, so "
                f"it has been estimated instead: {note_c}. The result is "
                "marked with a squiggle and carries a bigger margin of "
                "error to match."))
            return

        # 3) fall back to the constructed line itself
        if seed_dev_mm > INFER_MAX_SEED_DEV_MM:
            rep.notes.append(
                f"working it out from the other three edges puts it "
                f"{seed_dev_mm:.1f}mm from where the card was found, more "
                f"than the {INFER_MAX_SEED_DEV_MM:.0f}mm allowed")
            return
        strict_notes = "; ".join(rep.notes) or "refused"
        new_rep = EdgeFitReport(
            edge=side, method="inferred", n_points=0, n_rejected=0,
            rms_residual_px=None,
            angle_deg=inf.line.angle_from_nominal_deg(),
            bow_px=None, status="estimated",
            notes=[f"could not be measured: {strict_notes}", inf.note])
        res.edge_fits[res.edge_fits.index(rep)] = new_rep
        reports[side], lines[side] = new_rep, inf.line
        methods[side] = "inferred"
        est_extra[side] = inf.extra_unc_mm
        same = "left-to-right" if ax_p == "y" else "top-to-bottom"
        other = "top-to-bottom" if ax_p == "y" else "left-to-right"
        qa.append(QAFlag(
            "EDGE_INFERRED",
            f"The {side} edge could not be seen well enough to measure, so "
            f"its position was {inf.note}. The {other} result does not "
            f"use it; the {same} result does, so it is marked as an "
            "estimate.",
            severity="warning"))

    _frame_proximity_qa(qa, lines, Wimg, Himg,
                        extra=" In testing this moved results by about "
                              "0.1mm.")

    # --- render match (only when the card is known) ---
    card, Hpr, n_inl, med_err = None, None, 0, 0.0
    Hr = Wr = ppm_r = None
    if card_id:
        render_gray, render_rgb, url, card = render_source.get_render(card_id)
        Hr, Wr = render_gray.shape
        pad = int(0.02 * min(Wimg, Himg))
        mask = np.zeros_like(gray, np.uint8)
        mask[max(0, y0 - pad):min(Himg, y1 + pad),
             max(0, x0 - pad):min(Wimg, x1 + pad)] = 255
        Hpr, n_inl, med_err = match_to_render(gray, render_gray,
                                              photo_mask=mask)
        ppm_r = Wr / game.card_w_mm  # +-2% (render crop inside trim), differential use only
        res.render = RenderMatchReport(
            source="lorcanajson/ravensburger", url=url, render_size=(Wr, Hr),
            n_inliers=n_inl, median_reproj_px=med_err,
            notes=[f"card: {card.get('fullIdentifier', card_id)}"])
    else:
        res.method = "ink_hexagon"
    if card_id and n_inl < 200:
        qa.append(QAFlag("WEAK_RENDER_MATCH",
                         f"Only {n_inl} points could be matched between "
                         "your photo and the official card picture, where "
                         "a good match finds 600 to 800. The two may not be "
                         "lined up correctly.", severity="warning"))
    if card_id and med_err > 2.0:
        qa.append(QAFlag("HIGH_REPROJECTION_ERROR",
                         "Your photo and the official card picture line up "
                         f"to about {med_err:.1f} pixels, where 1 pixel is "
                         "normal. The alignment is loose."))

    # --- leaning-edge check, then the fourth edge ---
    # Mapped into the official picture, the photo's perspective is undone,
    # so the four cut edges of a real (rectangular) card must share one
    # rotation. An edge that leans away from the others was found on
    # something else - a case edge, a shadow, a glare streak - even when
    # its readings line up neatly. It is dropped, which lets the
    # fourth-edge path look for the real one.
    def _drop(side, note, code, msg):
        rep = reports[side]
        new_rep = EdgeFitReport(
            edge=side, method=rep.method, n_points=rep.n_points,
            n_rejected=rep.n_rejected, rms_residual_px=rep.rms_residual_px,
            angle_deg=rep.angle_deg, bow_px=rep.bow_px, status="refused",
            notes=list(rep.notes) + [note])
        res.edge_fits[res.edge_fits.index(rep)] = new_rep
        reports[side], lines[side] = new_rep, None
        est_extra.pop(side, None)
        qa.append(QAFlag(code, msg, severity="warning"))

    def _cut_rescue(side, inf):
        line = inf.line
        us = np.linspace(line.u_range[0], line.u_range[1], n_scans)
        mid = float(np.median(us))
        fits = []
        for w in CUT_RESCUE_WINDOWS_MM:
            try:
                cu, cv, cd = E.cut_scan(gray, side, line.v_at, us, inf.ppm,
                                        win_out_mm=w, win_in_mm=w,
                                        plateau_mm=-(w - 0.5))
            except ValueError:
                return None
            if cd.n_ok < max(15, 0.3 * n_scans):
                return None
            f = G.FittedLine.fit(line.orientation, cu, cv)
            if f.rms > CUT_RESCUE_RMS_PX or \
                    (max(cu) - min(cu)) < 0.3 * (us[-1] - us[0]):
                return None
            fits.append(f)
        pos = [float(f.v_at(mid)) for f in fits]
        spread_mm = (max(pos) - min(pos)) / inf.ppm
        if spread_mm > CUT_RESCUE_AGREE_MM:
            return None
        d_mm = abs(pos[0] - float(line.v_at(mid))) / inf.ppm
        if d_mm > RESCAN_K * inf.extra_unc_mm:
            return None
        best = fits[0]
        extra = math.sqrt(BASE_SYSTEMATIC_MM ** 2 + spread_mm ** 2
                          + (best.rms / math.sqrt(best.n) / inf.ppm) ** 2)
        if extra >= inf.extra_unc_mm:
            return None
        note = (f"read with a second kind of edge detector from "
                f"{best.n} readings, close to where the other edges said "
                f"it must be ({d_mm:.2f}mm away) and the same from two "
                f"different search widths (to {spread_mm:.2f}mm); an extra "
                f"{extra:.2f}mm has been added to the margin of error")
        return best, extra, note

    tried_fourth, dropped_leaning, no_rescan = set(), set(), set()
    for _ in range(4):
        changed = False
        if n_inl >= LEAN_MIN_INLIERS:
            leaning = _lean_check(Hpr, lines, methods)
            for side, why in leaning.items():
                if side in tried_fourth:
                    # it was itself found on a second look: build it
                    # instead, without looking a third time
                    tried_fourth.discard(side)
                    no_rescan.add(side)
                dropped_leaning.add(side)
                _drop(side, why, "EDGE_LEANING",
                      f"The {side} edge that was found {why}. It was not "
                      "used.")
                changed = True
            if leaning:
                # a constructed edge copied the geometry of a dropped one
                for side in _SIDES:
                    if methods.get(side) == "inferred" and \
                            lines[side] is not None:
                        qa[:] = [q for q in qa if not (
                            q.code == "EDGE_INFERRED"
                            and f"The {side} edge" in q.message)]
                        methods[side] = "step"
                        tried_fourth.discard(side)
                        _drop(side, "withdrawn: it was worked out from an "
                              "edge that turned out to be wrong",
                              "EDGE_INFERENCE_WITHDRAWN",
                              f"The {side} edge had been worked out from "
                              "the other edges, but one of those turned "
                              "out to be wrong, so that was withdrawn too.")
        for side in _SIDES:
            if lines[side] is None and side not in tried_fourth:
                tried_fourth.add(side)
                _fourth_edge(side, force_rescan=side in dropped_leaning,
                             no_rescan=side in no_rescan)
                changed = changed or lines[side] is not None
        if not changed:
            break
    missing = [s for s in _SIDES if lines[s] is None]
    if missing:
        for ax, (a, b) in {"x": ("left", "right"), "y": ("top", "bottom")}.items():
            bad = [s for s in (a, b) if lines[s] is None]
            if bad:
                res.shift_mm[ax] = Measurement.refused(
                    "mm", " ".join(
                        f"The {s} edge of the card could not be measured: "
                        f"{'; '.join(reports[s].notes)}." for s in bad),
                    RESHOOT)
            else:
                res.shift_mm[ax] = Measurement.refused(
                    "mm", f"The {a} and {b} edges were found, but all four "
                    "edges are needed to line the card up against the "
                    f"official picture, and the "
                    f"{' and '.join(missing)} could not be measured.",
                    RESHOOT)
        res.equivalent_ratio_lr = Ratio.refused(
            "LR", "Not all four edges of the card could be found.", RESHOOT)
        res.equivalent_ratio_tb = Ratio.refused(
            "TB", "Not all four edges of the card could be found.", RESHOOT)
        res.tilt.corrected = False
        return res

    # --- tilt from the physical quad ---
    quad = G.corner_quad(lines["left"], lines["right"],
                         lines["top"], lines["bottom"])
    kw, kh = G.keystone(quad)
    res.tilt.keystone_w_pct, res.tilt.keystone_h_pct = kw, kh
    cal = G.self_calibrate_tilt(quad, game.card_w_mm, game.card_h_mm,
                                (Wimg, Himg))
    if cal is not None:
        _, f_eq, pitch, yaw, total = cal
        res.tilt.focal_mm_equiv = f_eq
        res.tilt.pitch_deg, res.tilt.yaw_deg, res.tilt.total_deg = pitch, yaw, total
    else:
        res.tilt.notes.append(
            "the photo is square-on enough that the exact tilt angle "
            "cannot be worked out, and no correction was needed")
    res.corner_angles_deg = G.corner_angles(quad)
    wt = float(np.linalg.norm(quad[1] - quad[0]))
    wb = float(np.linalg.norm(quad[2] - quad[3]))
    hl = float(np.linalg.norm(quad[3] - quad[0]))
    hr = float(np.linalg.norm(quad[2] - quad[1]))
    res.aspect_ratio_measured = (hl + hr) / (wt + wb)
    nominal_aspect = game.card_h_mm / game.card_w_mm
    aspect_dev = abs(res.aspect_ratio_measured - nominal_aspect) / nominal_aspect
    if aspect_dev > 0.012:
        qa.append(QAFlag("ASPECT_DEVIATION",
                         "The card came out the wrong shape: "
                         f"{res.aspect_ratio_measured:.3f} times taller "
                         "than it is wide, where a real card is "
                         f"{nominal_aspect:.3f}. Check it is out of its "
                         "sleeve and flat, and look at the overlay picture "
                         "to see whether an edge was found in the wrong "
                         "place."))
    width_px = lines["right"].v_at(rows) - lines["left"].v_at(rows)
    inp.px_per_mm = float(np.median(width_px)) / game.card_w_mm
    wvar = float((width_px.max() - width_px.min()) / width_px.mean())

    # --- aspect sanity gate: a grossly non-card-shaped quad means at least
    # one edge latched onto glare/shadow/artwork; any shift computed from it
    # would be silently wrong, so refuse rather than estimate ---
    if aspect_dev > 0.03:
        reason = (f"The shape found is {aspect_dev * 100:.1f}% off a real "
                  f"card's shape ({res.aspect_ratio_measured:.3f} tall for "
                  f"every 1 across, against {nominal_aspect:.3f}). At least "
                  "one edge has been found badly wrong - usually a band of "
                  "glare, a shadow, or the edge of the artwork being "
                  "mistaken for the edge of the card. No number is given, "
                  "rather than a wrong one.")
        res.shift_mm = {"x": Measurement.refused("mm", reason, RESHOOT),
                        "y": Measurement.refused("mm", reason, RESHOOT)}
        res.equivalent_ratio_lr = Ratio.refused("LR", reason, RESHOOT)
        res.equivalent_ratio_tb = Ratio.refused("TB", reason, RESHOOT)
        return res

    # --- hybrid cut cross-check (shadow-band detection; QA only) ---
    # (an inferred edge was never read off the photo, so there is no
    # detection for the cross-check to disagree with)
    _shadow_band_qa(qa, gray,
                    {k: v for k, v in lines.items()
                     if methods.get(k) != "inferred"},
                    rows, cols, inp.px_per_mm)

    # --- the ink-cost hexagon: the result when the card is not known, a
    # cross-check when it is ---
    db_dir = getattr(render_source, "local_db_dir", None)
    if db_dir is None:
        from .games.lorcana import _default_local_db_dir
        db_dir = _default_local_db_dir()
    hexrep = measure_in_card(rgb, quad, game, inp.px_per_mm, lines, methods,
                             est_extra, card=card, db_dir=db_dir)
    res.hex_check = hexrep
    if hexrep.status == "measured":
        tol = 100 * game.hex_anchor.scale_tol
        for ax, name in (("x", "across"), ("y", "down")):
            dev = hexrep.size_vs_card_pct[ax]
            if abs(dev) <= tol:
                continue
            qa.append(QAFlag(
                "HEX_SCALE_MISMATCH",
                f"Measured {name}, the ink-cost hexagon is {dev:+.1f}% off "
                "the size the card's edges say it should be, where within "
                "1% is normal. One of the card's edges is probably in the "
                "wrong place, or the card is in a sleeve."))
            if not card_id:
                why = (f"the hexagon is {dev:+.1f}% off the size the card's "
                       "edges say it should be, so an edge is in the wrong "
                       "place")
                hexrep.shift_mm[ax] = Measurement.refused(
                    "mm", "The card was not identified, so the print shift "
                    f"has to come from the ink-cost hexagon, and {why}.",
                    RESHOOT)
    if not card_id:
        return _finish_from_hexagon(res, hexrep, game, rgb, lines,
                                    photo, out_dir, make_overlay, quad,
                                    inp.px_per_mm)

    # --- map physical edges into render space ---
    rlines = {}
    for side in _SIDES:
        pts_r = G.transform_points(Hpr, lines[side].points(60))
        orientation = "v" if side in ("left", "right") else "h"
        u, v = (pts_r[:, 1], pts_r[:, 0]) if orientation == "v" else \
               (pts_r[:, 0], pts_r[:, 1])
        rlines[side] = G.FittedLine.fit(orientation, u, v)

    mid_y, mid_x = Hr / 2.0, Wr / 2.0
    x_l = float(rlines["left"].v_at(mid_y))
    x_r = float(rlines["right"].v_at(mid_y))
    y_t = float(rlines["top"].v_at(mid_x))
    y_b = float(rlines["bottom"].v_at(mid_x))
    res.per_side_offsets_mm = {
        "left_outside_render": round(-x_l / ppm_r, 3),
        "right_outside_render": round((x_r - Wr) / ppm_r, 3),
        "top_outside_render": round(-y_t / ppm_r, 3),
        "bottom_outside_render": round((y_b - Hr) / ppm_r, 3),
    }

    # --- render-span physical plausibility gate ---
    span_bad = {}
    bounds = getattr(game, "render_span_bounds_mm", None)
    if bounds:
        span_bad = _render_span_violations(
            res.per_side_offsets_mm, bounds,
            {k: 2.0 * v for k, v in est_extra.items()})
        for ax, why in sorted(span_bad.items()):
            qa.append(QAFlag(
                "RENDER_SPAN_MISMATCH",
                f"Compared with the official card picture, {why}. That "
                "cannot be where the card was really cut, so no "
                + ("left-to-right" if ax == "x" else "top-to-bottom")
                + " print position is given. Something along that edge - a "
                "shadow, a band of glare, or a curled corner - was "
                "mistaken for the edge of the card.",
                severity="warning"))

    # shift of print relative to card: positive x = print displaced toward
    # the RIGHT card edge (card centre left of print centre).
    # Raw values measure true_shift + render_crop_bias; subtract the
    # calibrated per-axis bias (see GameSpec).
    xc = 0.5 * (x_l + x_r)
    yc = 0.5 * (y_t + y_b)
    bias = getattr(game, "render_crop_bias_mm", None) or {"x": 0.0, "y": 0.0}
    bias_unc = getattr(game, "render_crop_bias_unc_mm", None) or \
        {"x": 0.03, "y": 0.03}
    shift_x = (mid_x - xc) / ppm_r - bias["x"]
    shift_y = (mid_y - yc) / ppm_r - bias["y"]
    if bias["x"] or bias["y"]:
        res.qa.append(QAFlag(
            "RENDER_CROP_BIAS_CORRECTED",
            "The official card picture is cropped very slightly "
            "off-centre, by a known amount. That has been taken off the "
            f"result ({bias['x']:+.2f}mm across, {bias['y']:+.2f}mm down), "
            f"and the {bias_unc['y']:.2f}mm of doubt left over is included "
            "in the plus-or-minus figure.",
            severity="info"))

    def shift_unc(a_side, b_side) -> Uncertainty:
        stat_edges_r = math.sqrt(
            (rlines[a_side].rms ** 2) / max(rlines[a_side].n, 1)
            + (rlines[b_side].rms ** 2) / max(rlines[b_side].n, 1)) / 2.0
        align = med_err / math.sqrt(max(n_inl, 1)) * 3.0  # conservative
        stat = math.sqrt(stat_edges_r ** 2 + align ** 2) / ppm_r
        persp = abs(shift_x if a_side == "left" else shift_y) * wvar / 2.0 + 0.005
        # Face-aware cut definition: this path always measures a FRONT.
        # Per side, because the rim is not the same feature on opposite
        # edges of a full-art face (dark one side, bright the other), so
        # the two sides' errors are composed as independent rather than
        # cancelling - see GameSpec.cut_def_mm.
        axis = "x" if a_side == "left" else "y"
        ed_a = game.edge_def_mm("front", methods.get(a_side, "step"), axis,
                                inp.px_per_mm)
        ed_b = game.edge_def_mm("front", methods.get(b_side, "step"), axis,
                                inp.px_per_mm)
        ed = math.hypot(ed_a, ed_b) / 2.0
        b_unc = bias_unc["x" if a_side == "left" else "y"]
        est_u = math.sqrt(est_extra.get(a_side, 0.0) ** 2
                          + est_extra.get(b_side, 0.0) ** 2)
        edge_def = math.sqrt(ed ** 2 + b_unc ** 2 + est_u ** 2)
        return Uncertainty(statistical=stat, perspective=persp,
                           edge_definition=edge_def)

    res.shift_mm = {
        "x": Measurement(shift_x, "mm", shift_unc("left", "right")),
        "y": Measurement(shift_y, "mm", shift_unc("top", "bottom")),
    }
    for ax in span_bad:
        res.shift_mm[ax] = Measurement.refused(
            "mm", "The card edges found do not fit against the official "
            "card picture: " + span_bad[ax] + ". At least one edge is in "
            "the wrong place.", RESHOOT)

    # --- estimate tier: label and cap ---
    for ax, (a, b) in (("x", ("left", "right")), ("y", ("top", "bottom"))):
        if a not in est_extra and b not in est_extra:
            continue
        m = res.shift_mm[ax]
        if m.status != "measured":
            continue
        tot = m.uncertainty.total
        which = "/".join(s for s in (a, b) if s in est_extra)
        if tot > CAP_MM:
            res.shift_mm[ax] = Measurement.refused(
                "mm", f"The {which} edge could only be estimated, and the "
                f"estimate is good to no better than {tot:.2f}mm - past "
                f"the {CAP_MM:.1f}mm limit for a number worth quoting.",
                "A cleaner photo usually fixes this: card flat and "
                "unsleeved on plain white paper, soft light from all "
                "sides, sharply in focus.")
        else:
            m.status = "estimated"

    _hex_cross_check(res, hexrep, game)

    # --- grading-style equivalent ratios (convention, not measurement) ---
    def equiv(axis, shift, total_margin):
        # shift>0 = print toward right/bottom edge = right/bottom margin smaller
        a = total_margin / 2.0 + shift  # left/top border equivalent
        b = total_margin / 2.0 - shift
        if a <= 0 or b <= 0:
            return Ratio.refused(
                axis, "The print is shifted further than a normal border "
                "is wide, so a centring ratio would not mean anything "
                "here. Use the millimetre figure instead.")
        m = res.shift_mm["x" if axis == "LR" else "y"]
        s = m.uncertainty
        # d(pct)/d(shift) = 100 * 2 /? : pct = 100*a/(a+b), a+b const => 100/total
        k = 100.0 / total_margin
        unc = Uncertainty(statistical=s.statistical * k,
                          perspective=s.perspective * k,
                          edge_definition=s.edge_definition * k)
        return Ratio(axis=axis, first_pct=100.0 * a / (a + b),
                     uncertainty_pts=unc, status=m.status)

    res.equivalent_ratio_lr = (
        Ratio.refused("LR", res.shift_mm["x"].refusal_reason,
                      res.shift_mm["x"].refusal_advice)
        if res.shift_mm["x"].status == "refused"
        else equiv("LR", shift_x, game.equiv_margin_lr_mm))
    res.equivalent_ratio_tb = (
        Ratio.refused("TB", res.shift_mm["y"].refusal_reason,
                      res.shift_mm["y"].refusal_advice)
        if res.shift_mm["y"].status == "refused"
        else equiv("TB", shift_y, game.equiv_margin_tb_mm))

    # --- overlay: fitted edges + render bounds projected into the photo ---
    if make_overlay:
        import cv2
        ov = Overlay(rgb)
        for side in _SIDES:
            ov.line(lines[side], C_EDGE)
        Hrp = np.linalg.inv(Hpr)
        rect = np.array([[0, 0], [Wr, 0], [Wr, Hr], [0, Hr]], np.float64)
        rect_p = G.transform_points(Hrp, rect)
        cv2.polylines(ov.img, [rect_p.astype(np.int32)], True, C_FRAME, ov.lw)
        sx, sy = res.shift_mm["x"], res.shift_mm["y"]

        def _fmt(m):
            if m.value is None:
                return "not measured"
            pre = "~" if m.status == "estimated" else ""
            return f"{pre}{m.value:+.2f}mm"
        ov.banner([
            f"FRONT print shift: x {_fmt(sx)}  y {_fmt(sy)}",
            f"(+x = print sits toward the right edge, +y = toward bottom)",
            f"same as L/R {res.equivalent_ratio_lr.display or 'not measured'}"
            f"  T/B {res.equivalent_ratio_tb.display or 'not measured'}",
            f"matched to official picture: {n_inl} points, "
            f"{med_err:.2f}px apart"])
        if hexrep.status == "measured":
            _draw_hex_in_photo(ov, hexrep, quad, game, inp.px_per_mm)
        out = Path(out_dir) if out_dir else Path(photo).parent
        out.mkdir(parents=True, exist_ok=True)
        res.overlay = ov.save(out / (Path(photo).stem + "_front_overlay.jpg"))

    return res


# --- ink-cost hexagon: cross-check and card-unknown result ---------------

# the hexagon and the official picture disagree by more than this many
# sigmas (of the terms they do not share) before it is worth saying
HEX_AGREE_SIGMAS = 2.5


def _hex_cross_check(res, hexrep, game):
    """Compare the hexagon's shift with the official-picture shift."""
    if hexrep.status != "measured":
        res.qa.append(QAFlag(
            "HEX_CHECK_SKIPPED",
            "The ink-cost hexagon check could not be run: "
            f"{hexrep.refusal_reason}.", severity="info"))
        return
    bias_unc = game.render_crop_bias_unc_mm or {}
    agree, bad = {}, []
    for ax, name in (("x", "left to right"), ("y", "top to bottom")):
        a, b = hexrep.shift_mm[ax], res.shift_mm.get(ax)
        if b is None or b.status == "refused" or a.status == "refused":
            continue
        d = a.value - b.value
        agree[ax] = d
        common = bias_unc.get(ax, 0.0)
        sig = math.sqrt(max(a.uncertainty.total ** 2
                            + b.uncertainty.total ** 2
                            - 2 * common ** 2, 1e-6))
        if abs(d) > HEX_AGREE_SIGMAS * sig:
            bad.append(f"{name} they are {abs(d):.2f}mm apart, where up to "
                       f"{HEX_AGREE_SIGMAS * sig:.2f}mm is normal")
    hexrep.agreement_mm = agree or None
    if bad:
        res.qa.append(QAFlag(
            "HEX_ANCHOR_DISAGREES",
            "Two independent ways of measuring the print shift - matching "
            "the official card picture, and where the ink-cost hexagon "
            "sits - disagree: " + "; ".join(bad) + ". The hexagon only "
            "uses the top and left edges, so one of the card's edges is "
            "probably in the wrong place."))


def _finish_from_hexagon(res, hexrep, game, rgb, lines, photo,
                         out_dir, make_overlay, quad, ppm):
    """Card not known: the hexagon's shift is the result."""
    res.render = None
    if hexrep.status != "measured":
        reason = ("The card was not identified, so the print shift has to "
                  "come from the ink-cost hexagon, and "
                  f"{hexrep.refusal_reason}.")
        advice = ("Tell the analyser which card it is, or re-shoot flat and "
                  "in focus so the hexagon in the top-left corner is sharp.")
        res.shift_mm = {"x": Measurement.refused("mm", reason, advice),
                        "y": Measurement.refused("mm", reason, advice)}
        res.equivalent_ratio_lr = Ratio.refused("LR", reason, advice)
        res.equivalent_ratio_tb = Ratio.refused("TB", reason, advice)
        return res
    res.qa.append(QAFlag(
        "HEX_ANCHOR_USED",
        "The card was not identified, so the print shift comes from where "
        "the ink-cost hexagon sits against the top and left edges ("
        f"{hexrep.centre_from_left_mm:.2f}mm and "
        f"{hexrep.centre_from_top_mm:.2f}mm; on a centred card "
        f"{hexrep.expected_from_left_mm:.2f}mm and "
        f"{hexrep.expected_from_top_mm:.2f}mm).", severity="info"))
    for note in hexrep.notes:
        res.qa.append(QAFlag("HEX_LAYOUT_ASSUMED", note, severity="info"))
    res.shift_mm = dict(hexrep.shift_mm)
    for ax in ("x", "y"):
        m = res.shift_mm[ax]
        if m.status == "estimated" and m.uncertainty.total > CAP_MM:
            res.shift_mm[ax] = Measurement.refused(
                "mm", f"An edge this depends on could only be estimated, "
                f"and the estimate is good to no better than "
                f"{m.uncertainty.total:.2f}mm - past the {CAP_MM:.1f}mm "
                "limit for a number worth quoting.", RESHOOT)
    res.equivalent_ratio_lr = equivalent_ratio(
        "LR", res.shift_mm["x"], game.equiv_margin_lr_mm)
    res.equivalent_ratio_tb = equivalent_ratio(
        "TB", res.shift_mm["y"], game.equiv_margin_tb_mm)
    if make_overlay:
        ov = Overlay(rgb)
        for side in _SIDES:
            ov.line(lines[side], C_EDGE)
        _draw_hex_in_photo(ov, hexrep, quad, game, ppm)

        def _fmt(m):
            if m.value is None:
                return "not measured"
            return f"{'~' if m.status == 'estimated' else ''}{m.value:+.2f}mm"
        ov.banner([
            f"FRONT print shift: x {_fmt(res.shift_mm['x'])}  "
            f"y {_fmt(res.shift_mm['y'])}",
            "(+x = print sits toward the right edge, +y = toward bottom)",
            f"same as L/R {res.equivalent_ratio_lr.display or 'not measured'}"
            f"  T/B {res.equivalent_ratio_tb.display or 'not measured'}",
            f"from the {hexrep.layout} ink hexagon: x = where it is, "
            "+ = where it sits on a centred card"])
        out = Path(out_dir) if out_dir else Path(photo).parent
        out.mkdir(parents=True, exist_ok=True)
        res.overlay = ov.save(out / (Path(photo).stem + "_front_overlay.jpg"))
    return res


def _draw_hex_in_photo(ov, hexrep, quad, game, ppm):
    """Fitted hexagon centre, and where it would be on a centred card."""
    import cv2
    Hinv = np.linalg.inv(G.homography_to_card(quad, game.card_w_mm,
                                              game.card_h_mm))
    pts = G.transform_points(Hinv, np.array([
        [hexrep.centre_from_left_mm, hexrep.centre_from_top_mm],
        [hexrep.expected_from_left_mm, hexrep.expected_from_top_mm]]))
    size = max(8, int(0.8 * ppm))
    cv2.drawMarker(ov.img, tuple(int(round(v)) for v in pts[0]), C_FRAME,
                   cv2.MARKER_TILTED_CROSS, size, ov.lw)
    cv2.drawMarker(ov.img, tuple(int(round(v)) for v in pts[1]), C_EDGE,
                   cv2.MARKER_CROSS, size, ov.lw)
