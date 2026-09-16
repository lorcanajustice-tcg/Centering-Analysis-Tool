"""Bordered-face pipeline (e.g. Lorcana card backs with the gold frame line)."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np

from . import edges as E
from . import geometry as G
from .games.base import GameSpec
from .imgio import load_photo
from .locate import background_uniformity, coarse_locate
from .overlay import C_EDGE, C_FRAME, Overlay
from .types import (BackResult, EdgeFitReport, Measurement, QAFlag, Ratio,
                    TiltReport, Uncertainty)
from .uncertainty import border_stat_sigma_px, compose_ratio_uncertainty
# shared fit/gate machinery lives in fitting.py; re-exported here for
# backward compatibility (historical import site)
from .fitting import (SHADOW_BAND_MM, _colour_edge,  # noqa: F401
                      _edge_report, _frame_proximity_qa, _hybrid_cross_check,
                      _prefer_fit, _shadow_band_qa, _thin_fit)

_SIDES = ("left", "right", "top", "bottom")

# Along-side spacing of the frame-line scan lines (see the frame-line
# section of analyze_back). TAG copy Y / copy T backs, L/R by line count:
# 55 -> 52.89 / 53.23, 101 -> 51.16 / 52.77, 151 -> 50.98 / 52.59,
# 201 -> 51.15 / 52.84. 151 lines on those sides is a 0.41mm pitch.
FRAME_SCAN_PITCH_MM = 0.4
# the drift check re-traces the line 1.5x coarser (0.6mm on these numbers,
# still in the settled range) on lines that do not coincide with the main
# grid; the difference goes into the statistical term
FRAME_DRIFT_PITCH_RATIO = 1.5

def analyze_back(photo: str | Path, game: GameSpec, out_dir: Optional[str] = None,
                 n_scans: int = 55, make_overlay: bool = True) -> BackResult:
    rgb, gray, inp = load_photo(photo)
    Himg, Wimg = gray.shape
    res = BackResult(kind="back", game=game.name, input=inp, tilt=TiltReport())
    qa = res.qa

    # --- background QA ---
    corners = background_uniformity(gray)
    vals = list(corners.values())
    if max(vals) > 1.8 * max(min(vals), 1e-6):
        qa.append(QAFlag(
            "BACKGROUND_NONUNIFORM",
            "The background is much brighter in some corners than others "
            f"(from {min(vals):.0f} to {max(vals):.0f} on a 0-255 "
            "brightness scale). The card's edges are harder to pick out in "
            "the brightest and the most shadowed parts."))

    # --- coarse localization ---
    # illumination-invariant colour planes for the chromaticity edge scan
    # (see fitting._colour_edge); None when the photo is greyscale
    chroma = E.chromaticity(rgb) if rgb is not None and rgb.ndim == 3 else None

    coarse, ppm0 = coarse_locate(gray, game.card_w_mm, game.card_h_mm)
    if ppm0 is None:
        for s in _SIDES:
            res.edge_fits.append(EdgeFitReport(
                edge=s, method="texture", status="refused",
                notes=[coarse[s].reason
                       or "the card could not be found in this photo"]))
        find_it = ("Shoot the card flat and unsleeved on plain white paper, "
                   "filling most of the frame, with soft light from all "
                   "sides. Then try again.")
        res.borders_mm = {s: Measurement.refused(
            "mm", "The card could not be found in this photo - "
            f"{coarse[s].reason}.", find_it) for s in _SIDES}
        res.ratio_lr = Ratio.refused(
            "LR", "The card could not be found in this photo.", find_it)
        res.ratio_tb = Ratio.refused(
            "TB", "The card could not be found in this photo.", find_it)
        res.tilt.corrected = False
        return res

    # --- fine scan spans ---
    if coarse["top"].pos is not None and coarse["bottom"].pos is not None:
        ys, ye = coarse["top"].pos, coarse["bottom"].pos
    else:
        ys, ye = 0.22 * Himg, 0.78 * Himg  # coarse grid band (card presence verified)
    if coarse["left"].pos is not None and coarse["right"].pos is not None:
        xs, xe = coarse["left"].pos, coarse["right"].pos
    else:
        xs, xe = 0.22 * Wimg, 0.78 * Wimg
    rows = np.linspace(ys + 0.15 * (ye - ys), ys + 0.85 * (ye - ys), n_scans)
    cols = np.linspace(xs + 0.15 * (xe - xs), xs + 0.85 * (xe - xs), n_scans)

    # --- fine edge scans ---
    lines, reports, methods = {}, {}, {}
    for side in _SIDES:
        if coarse[side].pos is None:
            rep = EdgeFitReport(edge=side, method="texture", status="refused",
                                notes=[coarse[side].reason])
            lines[side], reports[side] = None, rep
            methods[side] = "texture"
            res.edge_fits.append(rep)
            continue
        us = rows if side in ("left", "right") else cols

        def _win(mm):
            # scanners need a minimum profile length at any resolution
            return max(mm * ppm0, E.MIN_SCAN_HALF_WINDOW_PX)

        slop = coarse[side].slop_mm
        out_mm = slop if slop is not None else 6.0
        in_mm = slop if slop is not None else 3.0

        def _fine(method, side=side, us=us, out_mm=out_mm, in_mm=in_mm):
            fn = E.step_scan if method == "step" else E.texture_scan
            return fn(gray, side, coarse[side].pos, us,
                      search_out_px=_win(out_mm), search_in_px=_win(in_mm))

        # prefer the scanner that established the coarse position; on
        # smooth light backgrounds that is the brightness step. If the
        # primary scanner yields no usable line (too few points OR an
        # inconsistent fit), try the alternate scanner before refusing.
        method = coarse[side].method or "texture"
        u_ok, v_ok, diag = _fine(method)
        line, rep = _edge_report(side, method, u_ok, v_ok, diag)
        if _thin_fit(line, diag):
            # the primary scanner either failed or fitted a thin/noisy
            # line; ask the other detector and keep the better-supported
            # of the two rather than whichever ran first
            alt = "step" if method == "texture" else "texture"
            u2, v2, d2 = _fine(alt)
            line2, rep2 = _edge_report(side, alt, u2, v2, d2)
            line, rep, diag, method = _prefer_fit(
                (line, rep, diag, method), (line2, rep2, d2, alt))
        # a coloured background keeps its hue in the shadow it wears at the
        # cut; brightness does not. Where the colour signal exists and
        # covers the edge, it defines the cut.
        line, rep, diag, method = _colour_edge(
            chroma, side, coarse[side].pos, us, _win(out_mm), _win(in_mm),
            (line, rep, diag, method), qa, ppm0)
        methods[side] = method
        if line is not None and line.bow_px and line.bow_px > 3.0:
            qa.append(QAFlag("CURL_SUSPECTED",
                             f"The {side} edge is not straight - it curves "
                             f"by {line.bow_px:.1f} pixels along its length. "
                             "A bent card makes its borders measure wrong."))
        lines[side], reports[side] = line, rep
        res.edge_fits.append(rep)

    # --- frame-edge proximity (radial distortion is unmodelled) ---
    _frame_proximity_qa(qa, lines, Wimg, Himg)

    have_lr = lines["left"] is not None and lines["right"] is not None
    have_tb = lines["top"] is not None and lines["bottom"] is not None

    # --- scale ---
    wvar = 0.0
    ppm_rows = None
    if have_lr:
        width_px = lines["right"].v_at(rows) - lines["left"].v_at(rows)
        ppm_rows = width_px / game.card_w_mm
        ppm = float(np.median(ppm_rows))
        wvar = float((width_px.max() - width_px.min()) / width_px.mean())
    elif have_tb:
        height_px = lines["bottom"].v_at(cols) - lines["top"].v_at(cols)
        ppm = float(np.median(height_px / game.card_h_mm))
        wvar = float((height_px.max() - height_px.min()) / height_px.mean())
    else:
        ppm = ppm0
    inp.px_per_mm = ppm

    # --- hybrid cut cross-check (shadow-band detection; QA only) ---
    _shadow_band_qa(qa, gray, lines, rows, cols, ppm)

    # --- tilt / rectification ---
    Hmm = None
    if have_lr and have_tb:
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
        Hmm = G.homography_to_card(quad, game.card_w_mm, game.card_h_mm)
        res.corner_angles_deg = G.corner_angles(quad)
        wt = float(np.linalg.norm(quad[1] - quad[0]))
        wb = float(np.linalg.norm(quad[2] - quad[3]))
        hl = float(np.linalg.norm(quad[3] - quad[0]))
        hr = float(np.linalg.norm(quad[2] - quad[1]))
        res.aspect_ratio_measured = (hl + hr) / (wt + wb)
        nominal_aspect = game.card_h_mm / game.card_w_mm
        if abs(res.aspect_ratio_measured - nominal_aspect) / nominal_aspect > 0.012:
            qa.append(QAFlag("ASPECT_DEVIATION",
                             "The card came out the wrong shape: "
                             f"{res.aspect_ratio_measured:.3f} times taller "
                             "than it is wide, where a real card is "
                             f"{nominal_aspect:.3f}. Check it is out of its "
                             "sleeve and flat, and look at the overlay "
                             "picture to see whether an edge was found in "
                             "the wrong place."))
    else:
        res.tilt.corrected = False
        res.tilt.notes.append(
            "not all four edges were found, so tilt could only be partly "
            f"corrected. The card's width changes by {wvar*100:.2f}% across "
            "the photo, which is the most this can be affecting the result")
        if have_lr:
            res.tilt.keystone_h_pct = float(
                (width_px[-1] - width_px[0]) / width_px.mean() * 100.0)

    # --- frame lines ---
    fspec = game.back_frame
    flines = {}
    if fspec is None:
        raise ValueError(f"game {game.name} has no back frame spec")
    # The frame line is sampled on its own, denser grid: a fixed pitch in
    # mm along the side rather than the cut edges' n_scans. At 55 lines
    # (1.1mm apart on a long side) the few lines where the halftone dot
    # chain gives a clean peak bunch up along the side, and the fitted line
    # sits where that stretch puts it - on the TAG copy Y back, L/R 52.89 at
    # 55 lines against 51.16 / 50.98 / 51.15 at 101 / 151 / 201. The
    # answer settles once the pitch is ~0.6mm or finer.
    frame_grid = {}
    for side, (grid, dim) in (("left", (rows, game.card_h_mm)),
                              ("right", (rows, game.card_h_mm)),
                              ("top", (cols, game.card_w_mm)),
                              ("bottom", (cols, game.card_w_mm))):
        span_mm = (grid[-1] - grid[0]) / max(ppm, 1e-9)
        n_f = max(len(grid), int(math.ceil(span_mm / FRAME_SCAN_PITCH_MM)))
        frame_grid[side] = np.linspace(grid[0], grid[-1], n_f)
    flines_alt, res_drift = {}, {}
    for side in _SIDES:
        if lines[side] is None:
            flines[side] = None
            continue
        g = frame_grid[side]
        # second grid for the drift check: FRAME_DRIFT_PITCH_RATIO times
        # coarser and offset by half its own step, so no line is shared
        n_alt = max(8, int(round(len(g) / FRAME_DRIFT_PITCH_RATIO)))
        step_alt = (g[-1] - g[0]) / n_alt
        g_alt = g[0] + step_alt * (np.arange(n_alt) + 0.5)
        au, av, adiag = E.frame_peak_scan(gray, side, lines[side], g_alt, ppm,
                                          min_peak=fspec.min_peak,
                                          search_mm=fspec.search_mm)
        aline, _ = _edge_report(f"frame_{side}", "frame_peak", au, av, adiag)
        flines_alt[side] = aline
        us = g
        fu, fv, fdiag = E.frame_peak_scan(gray, side, lines[side], us, ppm,
                                          min_peak=fspec.min_peak,
                                          search_mm=fspec.search_mm)
        fline, frep = _edge_report(f"frame_{side}", "frame_peak", fu, fv, fdiag)
        flines[side] = fline
        res.edge_fits.append(frep)
        if fline is not None and fdiag.n_attempted and \
                fline.n < 0.5 * fdiag.n_attempted:
            # The printed line is a halftone dot chain crossed by decorative
            # structure; where fewer than half the scan lines find it, the
            # survivors are a POSITIONALLY BIASED subset. Measured
            # 2026-08-21 on the TAG scans this was a bias, not a variance
            # problem (two halves of one sparse sample agreed to 5um while
            # the answer moved 75um/side with the line count), so no
            # variance correction could express it. Since v0.4.0 the line
            # is traced on a dense fixed-pitch grid, where the answer has
            # settled, and the move between that grid and a coarser,
            # non-overlapping one is carried in the statistical term. The
            # flag stays because a thin trace is still worth knowing about.
            qa.append(QAFlag(
                "FRAME_LINE_SPARSE",
                f"The printed gold line along the {side} could only be "
                f"traced in {fline.n} of {fdiag.n_attempted} places "
                f"({fdiag.summary()}). Those places are not spread evenly "
                "along the line, so the line was also traced a second way "
                "and the difference between the two has been added to the "
                "plus-or-minus figure.",
                severity="warning"))

    # --- borders ---
    ed = game.edge_def_px

    def border(side) -> Measurement:
        # Cut edge: face-aware (see GameSpec.cut_def_mm - the back's cut is
        # through a plain border and carries no full-art rim term).
        # Frame line: a printed feature, detector-domain only.
        axis = "x" if side in ("left", "right") else "y"
        def_border_mm = math.hypot(
            game.edge_def_mm("back", methods.get(side, "texture"), axis, ppm),
            ed["frame_peak"] / ppm)
        if lines[side] is None:
            return Measurement.refused(
                "mm", f"The {side} edge of the card could not be measured: "
                + ("; ".join(reports[side].notes)
                   or "no edge was found there") + ".",
                "Re-shoot on plain white paper with soft light from all "
                "sides, so that every edge of the card stands out clearly.")
        if flines[side] is None:
            return Measurement.refused(
                "mm", "The printed gold line could not be found near the "
                f"{side} edge.",
                "Use a sharper, larger photo with even lighting - the gold "
                "line is thin, and needs detail to pick out.")
        u0 = max(lines[side].u_range[0], flines[side].u_range[0])
        u1 = min(lines[side].u_range[1], flines[side].u_range[1])
        us = np.linspace(u0, u1, 60)
        full = rows if side in ("left", "right") else cols
        cov = (u1 - u0) / max(full[-1] - full[0], 1e-9)
        if cov < 0.6:
            qa.append(QAFlag(
                "PARTIAL_EDGE_SPAN",
                f"The {side} border could only be measured along "
                f"{cov*100:.0f}% of its length - the background was too "
                "plain to read against anywhere else. The number given "
                "describes that part of the border."))
        def width_of(fl):
            if Hmm is not None:
                f_mm = G.transform_points(Hmm, fl.points(60))
                ax = 0 if side in ("left", "right") else 1
                ref = 0.0 if side in ("left", "top") else (
                    game.card_w_mm if ax == 0 else game.card_h_mm)
                return float(abs(np.mean(f_mm[:, ax]) - ref))
            sgn = 1.0 if side in ("left", "top") else -1.0
            gaps = sgn * (fl.v_at(us) - lines[side].v_at(us))
            if side in ("left", "right") and ppm_rows is not None:
                local = np.interp(us, rows, ppm_rows)
            else:
                local = ppm
            return float(np.mean(gaps / local))

        w = width_of(flines[side])
        stat_px = border_stat_sigma_px(lines[side].rms, lines[side].n,
                                       flines[side].rms, flines[side].n)
        stat_mm = stat_px / ppm
        # sampling drift: the same line traced on a second, coarser grid
        # that shares no scan lines with the first. Whatever the choice of
        # lines moves the answer by is carried, not hidden.
        if flines_alt.get(side) is not None:
            drift = abs(width_of(flines_alt[side]) - w)
            res_drift[side] = drift
            stat_mm = math.hypot(stat_mm, drift)
        unc = Uncertainty(statistical=stat_mm,
                          perspective=w * wvar / 2.0,
                          edge_definition=def_border_mm)
        return Measurement(w, "mm", unc)

    res.borders_mm = {s: border(s) for s in _SIDES}

    def ratio(first, second, axis) -> Ratio:
        a, b = res.borders_mm[first], res.borders_mm[second]
        pairs = [(n, m) for n, m in ((first, a), (second, b))
                 if m.status != "measured"]
        if pairs:
            # Name the borders that failed and let their own rows carry the
            # detail - repeating two long reasons here reads as noise.
            which = " and ".join(n for n, _ in pairs)
            plural = "borders" if len(pairs) > 1 else "border"
            return Ratio.refused(
                axis, f"The {which} {plural} could not be measured, and "
                "both are needed for this. See below for why.",
                next((m.refusal_advice for _, m in pairs if m.refusal_advice),
                     None))
        unc = compose_ratio_uncertainty(
            a.value, b.value,
            a.uncertainty.statistical, b.uncertainty.statistical,
            a.uncertainty.edge_definition, b.uncertainty.edge_definition,
            perspective_pts=wvar * 100.0 / 2.0)
        return Ratio(axis=axis, first_pct=100.0 * a.value / (a.value + b.value),
                     uncertainty_pts=unc)

    res.ratio_lr = ratio("left", "right", "LR")
    res.ratio_tb = ratio("top", "bottom", "TB")

    # --- overlay ---
    if make_overlay:
        ov = Overlay(rgb)
        for side in _SIDES:
            if lines[side] is not None:
                ov.line(lines[side], C_EDGE)
            if flines.get(side) is not None:
                ov.line(flines[side], C_FRAME)
        txt = []
        for s in _SIDES:
            m = res.borders_mm[s]
            txt.append(f"{s[0].upper()}:{m.value:.2f}mm" if m.status == "measured"
                       else f"{s[0].upper()}:not measured")
        rl, rt = res.ratio_lr, res.ratio_tb
        ov.banner([
            f"BACK L/R: {rl.display or 'not measured'}"
            + (f" +-{rl.uncertainty_pts.total:.1f}pts" if rl.uncertainty_pts else ""),
            f"     T/B: {rt.display or 'not measured'}"
            + (f" +-{rt.uncertainty_pts.total:.1f}pts" if rt.uncertainty_pts else ""),
            "  ".join(txt)])
        out = Path(out_dir) if out_dir else Path(photo).parent
        out.mkdir(parents=True, exist_ok=True)
        res.overlay = ov.save(out / (Path(photo).stem + "_back_overlay.jpg"))

    return res
