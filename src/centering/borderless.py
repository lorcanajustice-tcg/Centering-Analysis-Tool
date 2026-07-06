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
from .back import _edge_report, _shadow_band_qa
from .games.base import GameSpec
from .imgio import load_photo
from .locate import background_uniformity, card_component_bbox, coarse_locate
from .overlay import C_EDGE, C_FRAME, Overlay
from .types import (BorderlessResult, EdgeFitReport, Measurement, QAFlag,
                    Ratio, RenderMatchReport, TiltReport, Uncertainty)
from .render_match import match_to_render
from .uncertainty import compose_ratio_uncertainty

_SIDES = ("left", "right", "top", "bottom")

# NOTE (2026-07-03): the render crop is NOT symmetric about the print
# centre vertically. Per-axis bias and its systematic uncertainty now come
# from GameSpec.render_crop_bias_mm / render_crop_bias_unc_mm (calibrated;
# see games/lorcana.py). The historical 0.03mm allowance applied to both
# axes only ever validated the x axis.


def analyze_borderless(photo: str | Path, card_id: str, game: GameSpec,
                       render_source=None, out_dir: Optional[str] = None,
                       n_scans: int = 50, make_overlay: bool = True
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
    try:
        x0, y0, x1, y1 = card_component_bbox(gray)
    except RuntimeError as err:
        coarse, _ = coarse_locate(gray, game.card_w_mm, game.card_h_mm)
        bad = [s for s in _SIDES if coarse[s].pos is None]
        if bad:
            raise RuntimeError(
                f"card could not be localized: component segmentation failed "
                f"({err}); per-side coarse scans failed for "
                + "; ".join(f"{s}: {coarse[s].reason}" for s in bad)) from err
        x0, y0 = int(coarse["left"].pos), int(coarse["top"].pos)
        x1, y1 = int(coarse["right"].pos), int(coarse["bottom"].pos)
    ppm0 = (x1 - x0) / game.card_w_mm

    corners_bg = background_uniformity(gray)
    vals = list(corners_bg.values())
    if max(vals) > 1.8 * max(min(vals), 1e-6):
        qa.append(QAFlag("BACKGROUND_NONUNIFORM",
                         "background brightness varies strongly across the "
                         "frame; low-contrast regions are excluded from edge "
                         "scans"))

    # --- physical edges: sub-pixel brightness step ---
    rows = np.linspace(y0 + 0.15 * (y1 - y0), y0 + 0.85 * (y1 - y0), n_scans)
    cols = np.linspace(x0 + 0.15 * (x1 - x0), x0 + 0.85 * (x1 - x0), n_scans)
    approx = {"left": x0, "right": x1, "top": y0, "bottom": y1}
    lines, reports, methods = {}, {}, {}
    for side in _SIDES:
        us = rows if side in ("left", "right") else cols
        u_ok, v_ok, diag = E.step_scan(
            gray, side, approx[side], us,
            search_out_px=2.5 * ppm0, search_in_px=2.5 * ppm0)
        line, rep = _edge_report(side, "step", u_ok, v_ok, diag)
        methods[side] = "step"
        if line is None:
            # black border on a dark textured mat has no brightness step,
            # but the texture transition still marks the cut (the same
            # signal the back pipeline uses on dark mats)
            u2, v2, d2 = E.texture_scan(
                gray, side, approx[side], us,
                search_out_px=2.5 * ppm0, search_in_px=2.5 * ppm0)
            line2, rep2 = _edge_report(side, "texture", u2, v2, d2)
            if line2 is not None:
                line, rep, diag = line2, rep2, d2
                methods[side] = "texture"
        if line is not None and line.bow_px and line.bow_px > 3.0:
            qa.append(QAFlag("CURL_SUSPECTED",
                             f"{side} edge bows {line.bow_px:.1f}px over its "
                             "span; foil curl biases the measured cut position"))
        if diag.n_ok and diag.n_ok < 0.7 * diag.n_attempted:
            qa.append(QAFlag("EDGE_PARTIALLY_EXCLUDED",
                             f"{side} edge: {diag.n_attempted - diag.n_ok}/"
                             f"{diag.n_attempted} scan lines excluded "
                             f"({diag.summary()}); fit uses the clean regions"))
        lines[side], reports[side] = line, rep
        res.edge_fits.append(rep)

    margin = 0.05 * min(Wimg, Himg)
    for side, line in lines.items():
        if line is None:
            continue
        pts = line.points(20)
        if (pts[:, 0].min() < margin or pts[:, 0].max() > Wimg - margin or
                pts[:, 1].min() < margin or pts[:, 1].max() > Himg - margin):
            qa.append(QAFlag("RADIAL_DISTORTION_RISK",
                             f"{side} card edge lies within 5% of the photo "
                             "frame edge; radial lens distortion is not "
                             "modelled and can bias the result "
                             "(seen at the 0.1mm level in validation)"))

    missing = [s for s in _SIDES if lines[s] is None]
    if missing:
        for ax, (a, b) in {"x": ("left", "right"), "y": ("top", "bottom")}.items():
            bad = [s for s in (a, b) if lines[s] is None]
            if bad:
                res.shift_mm[ax] = Measurement.refused(
                    "mm", "; ".join(
                        f"{s} edge unmeasurable: "
                        f"{'; '.join(reports[s].notes)}" for s in bad))
            else:
                res.shift_mm[ax] = Measurement.refused(
                    "mm", f"{a}/{b} edges measured, but render alignment "
                    "requires the full physical quad "
                    f"({', '.join(missing)} unmeasurable)")
        res.equivalent_ratio_lr = Ratio.refused("LR", "physical edges incomplete")
        res.equivalent_ratio_tb = Ratio.refused("TB", "physical edges incomplete")
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
        res.tilt.notes.append("focal self-calibration degenerate "
                              "(near fronto-parallel); keystone reported")
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
                         f"measured H/W {res.aspect_ratio_measured:.4f} vs "
                         f"nominal {nominal_aspect:.4f}; check for sleeve, "
                         "curl or a mis-detected edge"))
    width_px = lines["right"].v_at(rows) - lines["left"].v_at(rows)
    inp.px_per_mm = float(np.median(width_px)) / game.card_w_mm
    wvar = float((width_px.max() - width_px.min()) / width_px.mean())

    # --- aspect sanity gate: a grossly non-card-shaped quad means at least
    # one edge latched onto glare/shadow/artwork; any shift computed from it
    # would be silently wrong, so refuse rather than estimate ---
    if aspect_dev > 0.03:
        reason = (f"measured quad H/W {res.aspect_ratio_measured:.4f} deviates "
                  f"{aspect_dev * 100:.1f}% from nominal {nominal_aspect:.4f}; "
                  "at least one edge is grossly mis-detected (glare band, "
                  "shadow or artwork boundary); refusing rather than "
                  "reporting a biased print shift")
        res.shift_mm = {"x": Measurement.refused("mm", reason),
                        "y": Measurement.refused("mm", reason)}
        res.equivalent_ratio_lr = Ratio.refused("LR", reason)
        res.equivalent_ratio_tb = Ratio.refused("TB", reason)
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
                         f"only {n_inl} RANSAC inliers (expect 600-800); "
                         "alignment may be unreliable", severity="warning"))
    if med_err > 2.0:
        qa.append(QAFlag("HIGH_REPROJECTION_ERROR",
                         f"median reprojection {med_err:.2f}px in render space "
                         "(expect ~1px)"))

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
            f"calibrated render-crop bias subtracted from the raw shift "
            f"(x {bias['x']:+.2f}mm, y {bias['y']:+.2f}mm; systematic "
            f"+-{bias_unc['y']:.2f}mm retained in the uncertainty)",
            severity="info"))

    def shift_unc(a_side, b_side) -> Uncertainty:
        stat_edges_r = math.sqrt(
            (rlines[a_side].rms ** 2) / max(rlines[a_side].n, 1)
            + (rlines[b_side].rms ** 2) / max(rlines[b_side].n, 1)) / 2.0
        align = med_err / math.sqrt(max(n_inl, 1)) * 3.0  # conservative
        stat = math.sqrt(stat_edges_r ** 2 + align ** 2) / ppm_r
        persp = abs(shift_x if a_side == "left" else shift_y) * wvar / 2.0 + 0.005
        ed_a = game.edge_def_px[methods.get(a_side, "step")]
        ed_b = game.edge_def_px[methods.get(b_side, "step")]
        ed = math.sqrt(ed_a ** 2 + ed_b ** 2) / 2.0 / inp.px_per_mm
        b_unc = bias_unc["x" if a_side == "left" else "y"]
        edge_def = math.sqrt(ed ** 2 + b_unc ** 2)
        return Uncertainty(statistical=stat, perspective=persp,
                           edge_definition=edge_def)

    res.shift_mm = {
        "x": Measurement(shift_x, "mm", shift_unc("left", "right")),
        "y": Measurement(shift_y, "mm", shift_unc("top", "bottom")),
    }

    # --- grading-style equivalent ratios (convention, not measurement) ---
    def equiv(axis, shift, total_margin):
        # shift>0 = print toward right/bottom edge = right/bottom margin smaller
        a = total_margin / 2.0 + shift  # left/top border equivalent
        b = total_margin / 2.0 - shift
        if a <= 0 or b <= 0:
            return Ratio.refused(axis, "shift exceeds the nominal margin; "
                                 "equivalence convention breaks down")
        m = res.shift_mm["x" if axis == "LR" else "y"]
        s = m.uncertainty
        # d(pct)/d(shift) = 100 * 2 /? : pct = 100*a/(a+b), a+b const => 100/total
        k = 100.0 / total_margin
        unc = Uncertainty(statistical=s.statistical * k,
                          perspective=s.perspective * k,
                          edge_definition=s.edge_definition * k)
        return Ratio(axis=axis, first_pct=100.0 * a / (a + b),
                     uncertainty_pts=unc)

    res.equivalent_ratio_lr = equiv("LR", shift_x, game.equiv_margin_lr_mm)
    res.equivalent_ratio_tb = equiv("TB", shift_y, game.equiv_margin_tb_mm)

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
        ov.banner([
            f"FRONT print shift: x {sx.value:+.2f}mm  y {sy.value:+.2f}mm",
            f"(+x = print toward right edge, +y = toward bottom)",
            f"equiv L/R {res.equivalent_ratio_lr.display or 'refused'}  "
            f"T/B {res.equivalent_ratio_tb.display or 'refused'}",
            f"render match: {n_inl} inliers, {med_err:.2f}px reproj"])
        out = Path(out_dir) if out_dir else Path(photo).parent
        out.mkdir(parents=True, exist_ok=True)
        res.overlay = ov.save(out / (Path(photo).stem + "_front_overlay.jpg"))

    return res
