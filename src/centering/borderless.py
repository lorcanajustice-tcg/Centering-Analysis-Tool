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
from .estimate import CAP_MM, rescue_edge
from .games.base import GameSpec
from .imgio import load_photo
from .locate import background_uniformity, card_component_bbox, coarse_locate
from .overlay import C_EDGE, C_FRAME, Overlay
from .types import (BorderlessResult, EdgeFitReport, Measurement, QAFlag,
                    Ratio, RenderMatchReport, TiltReport, Uncertainty)
from .render_match import match_to_render
from .uncertainty import compose_ratio_uncertainty

_SIDES = ("left", "right", "top", "bottom")

# The advice given whenever a photo simply is not good enough to measure.
RESHOOT = ("Re-shoot the card flat and unsleeved on plain white paper, with "
           "soft light coming from all sides, and try again.")

# NOTE (2026-07-03): the render crop is NOT symmetric about the print
# centre vertically. Per-axis bias and its systematic uncertainty now come
# from GameSpec.render_crop_bias_mm / render_crop_bias_unc_mm (calibrated;
# see games/lorcana.py). The historical 0.03mm allowance applied to both
# axes only ever validated the x axis.


def _render_span_violations(offsets_mm: dict, bounds: dict) -> dict:
    """Physical-plausibility check of the fitted quad against the render.

    The cut always lies OUTSIDE the render bounds (the render is cropped
    inside trim, crop >= 0) and the left+right / top+bottom render-to-cut
    totals are layout-locked render-crop constants plus the manufacture
    cut size - independent of where the cut landed on this card. A fitted
    edge dragged off the cut by a hard cast shadow or foil curl breaks
    these bounds even when its fit statistics look clean (a cast shadow's
    outer boundary is as sharp as a real cut and tonally continuous with
    dark card art). Returns {axis: reason}; empty when plausible.
    """
    out = {}
    s_lo, s_hi = bounds["side"]
    for ax, (a, b) in (("x", ("left", "right")), ("y", ("top", "bottom"))):
        oa = offsets_mm[f"{a}_outside_render"]
        ob = offsets_mm[f"{b}_outside_render"]
        t_lo, t_hi = bounds[f"{ax}_total"]
        probs = [f"the {n} edge sits {v:+.2f}mm outside the official "
                 f"picture, where every real card falls between "
                 f"{s_lo:+.2f} and {s_hi:+.2f}mm"
                 for n, v in ((a, oa), (b, ob)) if not s_lo <= v <= s_hi]
        if not t_lo <= oa + ob <= t_hi:
            probs.append(f"the {a} and {b} edges together sit "
                         f"{oa + ob:.2f}mm outside the official picture, "
                         f"where every real card falls between {t_lo:.2f} "
                         f"and {t_hi:.2f}mm")
        if probs:
            out[ax] = "; ".join(probs)
    return out


def analyze_borderless(photo: str | Path, card_id: str, game: GameSpec,
                       render_source=None, out_dir: Optional[str] = None,
                       n_scans: int = 50, make_overlay: bool = True,
                       manual_bbox: Optional[tuple] = None
                       ) -> BorderlessResult:
    rgb, gray, inp = load_photo(photo)
    Himg, Wimg = gray.shape
    res = BorderlessResult(kind="borderless", game=game.name, input=inp,
                           tilt=TiltReport())
    qa = res.qa

    if render_source is None:
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
                    "The card could not be found in this photo. "
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
    for side in _SIDES:
        us = rows if side in ("left", "right") else cols
        u_ok, v_ok, diag = E.step_scan(
            gray, side, approx[side], us,
            search_out_px=seed_slop_mm * ppm0,
            search_in_px=seed_slop_mm * ppm0)
        line, rep = _edge_report(side, "step", u_ok, v_ok, diag)
        methods[side] = "step"
        u2, v2, d2 = np.array([]), np.array([]), None
        if _thin_fit(line, diag):
            # a black border on a dark textured mat has no brightness step
            # at all, and a full-bleed face whose artwork happens to match
            # the surround leaves the step scanner clinging to a handful of
            # lines. The texture transition marks the same cut; keep
            # whichever fit the edge actually supports.
            u2, v2, d2 = E.texture_scan(
                gray, side, approx[side], us,
                search_out_px=seed_slop_mm * ppm0,
                search_in_px=seed_slop_mm * ppm0)
            line2, rep2 = _edge_report(side, "texture", u2, v2, d2)
            line, rep, diag, methods[side] = _prefer_fit(
                (line, rep, diag, "step"), (line2, rep2, d2, "texture"))
        # chromaticity edge: on a coloured background the shadow at the cut
        # is a brightness ramp but not a hue change, and a full-bleed face
        # can carry a dark cut-edge rim on one side and a bright one on the
        # other - both displace a brightness scan, asymmetrically.
        line, rep, diag, methods[side] = _colour_edge(
            chroma, side, approx[side], us,
            seed_slop_mm * ppm0, seed_slop_mm * ppm0,
            (line, rep, diag, methods[side]), qa, ppm0)
        if line is None:
            # estimate tier: the strict tier refused this edge; attempt an
            # explicitly-labelled rescue (cluster split + hybrid cut
            # cross-check arbitration). Never silent: the report keeps the
            # strict refusal, the edge fit is status="estimated", and the
            # axis result is capped at CAP_MM total uncertainty.
            strict_notes = "; ".join(rep.notes) or "refused"
            for mname, uu, vv in (("step", u_ok, v_ok), ("texture", u2, v2)):
                if len(uu) < 4:
                    continue
                est, why = rescue_edge(gray, side, uu, vv, us, ppm0)
                if est is None:
                    rep.notes.append(f"tried to estimate it instead, but "
                                     f"{why}")
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
                methods[side] = mname
                est_extra[side] = est.extra_unc_mm
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
        lines[side], reports[side] = line, rep
        res.edge_fits.append(rep)

    _frame_proximity_qa(qa, lines, Wimg, Himg,
                        extra=" In testing this moved results by about "
                              "0.1mm.")

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
    _shadow_band_qa(qa, gray, lines, rows, cols, inp.px_per_mm)

    # --- render match ---
    render_gray, render_rgb, url, card = render_source.get_render(card_id)
    Hr, Wr = render_gray.shape
    pad = int(0.02 * min(Wimg, Himg))
    mask = np.zeros_like(gray, np.uint8)
    mask[max(0, y0 - pad):min(Himg, y1 + pad),
         max(0, x0 - pad):min(Wimg, x1 + pad)] = 255
    Hpr, n_inl, med_err = match_to_render(gray, render_gray, photo_mask=mask)
    ppm_r = Wr / game.card_w_mm  # +-2% (render crop inside trim), differential use only
    res.render = RenderMatchReport(
        source="lorcanajson/ravensburger", url=url, render_size=(Wr, Hr),
        n_inliers=n_inl, median_reproj_px=med_err,
        notes=[f"card: {card.get('fullIdentifier', card_id)}"])
    if n_inl < 200:
        qa.append(QAFlag("WEAK_RENDER_MATCH",
                         f"Only {n_inl} points could be matched between "
                         "your photo and the official card picture, where "
                         "a good match finds 600 to 800. The two may not be "
                         "lined up correctly.", severity="warning"))
    if med_err > 2.0:
        qa.append(QAFlag("HIGH_REPROJECTION_ERROR",
                         "Your photo and the official card picture line up "
                         f"to about {med_err:.1f} pixels, where 1 pixel is "
                         "normal. The alignment is loose."))

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
        span_bad = _render_span_violations(res.per_side_offsets_mm, bounds)
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
        out = Path(out_dir) if out_dir else Path(photo).parent
        out.mkdir(parents=True, exist_ok=True)
        res.overlay = ov.save(out / (Path(photo).stem + "_front_overlay.jpg"))

    return res
