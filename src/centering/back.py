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

_SIDES = ("left", "right", "top", "bottom")


def _edge_report(name, method, us, vs, diag, flag_rms=1.5, min_pts=10):
    rep = EdgeFitReport(edge=name, method=method, n_points=diag.n_ok,
                        n_rejected=diag.n_attempted - diag.n_ok)
    if diag.n_ok < min_pts:
        rep.status = "refused"
        rep.notes.append(f"only {diag.n_ok}/{diag.n_attempted} scan lines "
                         f"usable ({diag.summary()})")
        return None, rep
    orientation = "v" if name.endswith(("left", "right")) else "h"
    line = G.FittedLine.fit(orientation, us, vs)
    rep.n_points = line.n
    rep.n_rejected += line.n_rej
    rep.rms_residual_px = line.rms
    rep.angle_deg = line.angle_from_nominal_deg()
    rep.bow_px = line.bow_px
    if line.rms > 4.0:
        rep.status = "refused"
        rep.notes.append(f"fit residual {line.rms:.2f}px far above target; "
                         "detections inconsistent")
        return None, rep
    if line.rms > flag_rms:
        rep.status = "flagged"
        rep.notes.append(f"fit residual {line.rms:.2f}px above {flag_rms}px target")
    if diag.n_ok < 0.7 * diag.n_attempted:
        rep.notes.append(f"partial coverage: {diag.summary()}")
    return line, rep


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
            "background brightness varies strongly across the frame "
            f"({ {k: round(v) for k, v in corners.items()} }); texture contrast "
            "and edge definition degrade in the bright/shadowed regions"))

    # --- coarse localization ---
    coarse, ppm0 = coarse_locate(gray, game.card_w_mm, game.card_h_mm)
    if ppm0 is None:
        for s in _SIDES:
            res.edge_fits.append(EdgeFitReport(
                edge=s, method="texture", status="refused",
                notes=[coarse[s].reason or "coarse localization failed"]))
        res.borders_mm = {s: Measurement.refused(
            "mm", f"card not localized: {coarse[s].reason}") for s in _SIDES}
        res.ratio_lr = Ratio.refused("LR", "card could not be localized in the photo")
        res.ratio_tb = Ratio.refused("TB", "card could not be localized in the photo")
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
    lines, reports = {}, {}
    for side in _SIDES:
        if coarse[side].pos is None:
            rep = EdgeFitReport(edge=side, method="texture", status="refused",
                                notes=[coarse[side].reason])
            lines[side], reports[side] = None, rep
            res.edge_fits.append(rep)
            continue
        us = rows if side in ("left", "right") else cols
        u_ok, v_ok, diag = E.texture_scan(
            gray, side, coarse[side].pos, us,
            search_out_px=6.0 * ppm0, search_in_px=3.0 * ppm0)
        line, rep = _edge_report(side, "texture", u_ok, v_ok, diag)
        if line is not None and line.bow_px and line.bow_px > 3.0:
            qa.append(QAFlag("CURL_SUSPECTED",
                             f"{side} edge bows {line.bow_px:.1f}px over its span; "
                             "foil curl biases border widths"))
        lines[side], reports[side] = line, rep
        res.edge_fits.append(rep)

    # --- frame-edge proximity (radial distortion is unmodelled) ---
    margin = 0.05 * min(Wimg, Himg)
    for side, line in lines.items():
        if line is None:
            continue
        pts = line.points(20)
        if (pts[:, 0].min() < margin or pts[:, 0].max() > Wimg - margin or
                pts[:, 1].min() < margin or pts[:, 1].max() > Himg - margin):
            qa.append(QAFlag("RADIAL_DISTORTION_RISK",
                             f"{side} card edge lies within 5% of the photo frame "
                             "edge; radial lens distortion is not modelled there"))

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
            res.tilt.notes.append("focal self-calibration degenerate "
                                  "(near fronto-parallel); keystone reported")
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
                             f"measured H/W {res.aspect_ratio_measured:.4f} vs nominal "
                             f"{nominal_aspect:.4f}; check for sleeve, curl or a "
                             "mis-detected edge"))
    else:
        res.tilt.corrected = False
        res.tilt.notes.append(
            "full rectification unavailable (missing edges); perspective "
            f"handled by per-row scale; span scale variation {wvar*100:.2f}% "
            "bounds the residual effect")
        if have_lr:
            res.tilt.keystone_h_pct = float(
                (width_px[-1] - width_px[0]) / width_px.mean() * 100.0)

    # --- frame lines ---
    fspec = game.back_frame
    flines = {}
    if fspec is None:
        raise ValueError(f"game {game.name} has no back frame spec")
    for side in _SIDES:
        if lines[side] is None:
            flines[side] = None
            continue
        us = rows if side in ("left", "right") else cols
        fu, fv, fdiag = E.frame_peak_scan(gray, side, lines[side], us, ppm,
                                          min_peak=fspec.min_peak,
                                          search_mm=fspec.search_mm)
        fline, frep = _edge_report(f"frame_{side}", "frame_peak", fu, fv, fdiag)
        flines[side] = fline
        res.edge_fits.append(frep)

    # --- borders ---
    ed = game.edge_def_px
    def_border_px = math.sqrt(ed["texture"] ** 2 + ed["frame_peak"] ** 2)

    def border(side) -> Measurement:
        if lines[side] is None:
            return Measurement.refused(
                "mm", f"{side} card edge unmeasurable: "
                f"{'; '.join(reports[side].notes) or 'edge scan failed'}")
        if flines[side] is None:
            return Measurement.refused(
                "mm", f"{side} frame line not detected inside search window")
        u0 = max(lines[side].u_range[0], flines[side].u_range[0])
        u1 = min(lines[side].u_range[1], flines[side].u_range[1])
        us = np.linspace(u0, u1, 60)
        full = rows if side in ("left", "right") else cols
        cov = (u1 - u0) / max(full[-1] - full[0], 1e-9)
        if cov < 0.6:
            qa.append(QAFlag(
                "PARTIAL_EDGE_SPAN",
                f"{side} border measured over only {cov*100:.0f}% of the "
                "scan span (weak background texture elsewhere); value "
                "represents that region"))
        if Hmm is not None:
            f_mm = G.transform_points(Hmm, flines[side].points(60))
            ax = 0 if side in ("left", "right") else 1
            ref = 0.0 if side in ("left", "top") else (
                game.card_w_mm if ax == 0 else game.card_h_mm)
            w = float(abs(np.mean(f_mm[:, ax]) - ref))
        else:
            sgn = 1.0 if side in ("left", "top") else -1.0
            gaps = sgn * (flines[side].v_at(us) - lines[side].v_at(us))
            if side in ("left", "right") and ppm_rows is not None:
                local = np.interp(us, rows, ppm_rows)
            else:
                local = ppm
            w = float(np.mean(gaps / local))
        stat_px = border_stat_sigma_px(lines[side].rms, lines[side].n,
                                       flines[side].rms, flines[side].n)
        unc = Uncertainty(statistical=stat_px / ppm,
                          perspective=w * wvar / 2.0,
                          edge_definition=def_border_px / ppm)
        return Measurement(w, "mm", unc)

    res.borders_mm = {s: border(s) for s in _SIDES}

    def ratio(first, second, axis) -> Ratio:
        a, b = res.borders_mm[first], res.borders_mm[second]
        bad = [m for m in (a, b) if m.status != "measured"]
        if bad:
            return Ratio.refused(axis, "; ".join(m.refusal_reason for m in bad))
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
                       else f"{s[0].upper()}:refused")
        rl, rt = res.ratio_lr, res.ratio_tb
        ov.banner([
            f"BACK L/R: {rl.display or 'refused'}"
            + (f" +-{rl.uncertainty_pts.total:.1f}pts" if rl.uncertainty_pts else ""),
            f"     T/B: {rt.display or 'refused'}"
            + (f" +-{rt.uncertainty_pts.total:.1f}pts" if rt.uncertainty_pts else ""),
            "  ".join(txt)])
        out = Path(out_dir) if out_dir else Path(photo).parent
        out.mkdir(parents=True, exist_ok=True)
        res.overlay = ov.save(out / (Path(photo).stem + "_back_overlay.jpg"))

    return res
