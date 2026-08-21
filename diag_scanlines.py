#!/usr/bin/env python3
"""Per-scanline edge-detection diagnostic for refused photos.

Reuses the production scanners line-by-line (bit-identical behaviour) so
every scan line's outcome -- accepted position or rejection reason -- can
be dumped and visualized. Use on photos the analyzer refuses to see
exactly WHERE and WHY each edge fails, and whether refused edges show a
two-cluster pattern (e.g. sleeve edge vs card cut).

Usage:
    python diag_scanlines.py PHOTO [--n 50] [--out DIR]

Writes <out>/report.txt, <out>/scanlines.csv, <out>/overlay.png
(default out: results/diag_<photo-stem>/). Never commit the outputs
(card imagery policy).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from centering import edges as E                       # noqa: E402
from centering.imgio import load_photo                 # noqa: E402
from centering.locate import (background_uniformity,   # noqa: E402
                              card_component_bbox, coarse_locate)
from centering.games.lorcana import LORCANA            # noqa: E402

SIDES = ("left", "right", "top", "bottom")

COLORS = {
    "ok": (0, 220, 0),
    "insufficient_contrast": (255, 60, 60),
    "insufficient_texture_contrast": (255, 60, 60),
    "glare_band_skipped": (255, 165, 0),
    "shadowed_outside_level": (255, 0, 255),
    "no_crossing": (150, 150, 150),
    "no_sustained_crossing": (150, 150, 150),
    "non_monotonic_at_edge": (120, 120, 255),
    "band_truncated": (80, 80, 80),
}
FALLBACK = (255, 255, 0)


def scan_one(scanner, gray, side, approx, u, so, si, **kw):
    """Run a production scanner on a single scan line.

    Returns (v_or_None, outcome_str)."""
    us, vs, diag = scanner(gray, side, approx, np.array([float(u)]),
                           search_out_px=so, search_in_px=si, **kw)
    if len(us) == 1:
        return float(vs[0]), "ok"
    reasons = list(diag.reject_reasons.keys())
    return None, (reasons[-1] if reasons else "unknown")


def per_line(scanner, gray, side, approx, us, so, si, **kw):
    out = []
    for u in us:
        v, why = scan_one(scanner, gray, side, approx, u, so, si, **kw)
        out.append((float(u), v, why))
    return out


def clusters(vals, gap_px=4.0):
    """Split sorted detections at gaps > gap_px; return list of arrays."""
    if not vals:
        return []
    v = np.sort(np.asarray(vals, dtype=float))
    cuts = np.where(np.diff(v) > gap_px)[0]
    return np.split(v, cuts + 1)


def fmt_clusters(vals, ppm):
    cl = clusters(vals)
    if not cl:
        return "  (no detections)"
    cl = sorted(cl, key=len, reverse=True)
    lines = []
    for c in cl:
        med = float(np.median(c))
        mad = float(np.median(np.abs(c - med)))
        lines.append(f"  cluster n={len(c):2d}  median={med:8.2f}px  "
                     f"mad={mad:5.2f}px")
    if len(cl) >= 2:
        sep = abs(float(np.median(cl[0])) - float(np.median(cl[1])))
        mm = f" = {sep / ppm:.2f}mm" if ppm else ""
        lines.append(f"  top-2 cluster separation: {sep:.1f}px{mm}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("photo")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", default=None)
    ap.add_argument("--bbox", default=None,
                    help="x0,y0,x1,y1 manual card box (image px); "
                    "overrides auto localization for the fine scans")
    args = ap.parse_args()

    rgb, gray, inp = load_photo(args.photo)
    H, W = gray.shape
    out_dir = Path(args.out) if args.out else (
        Path(__file__).resolve().parent / "results"
        / f"diag_{Path(args.photo).stem}")
    out_dir.mkdir(parents=True, exist_ok=True)

    rep = []
    say = rep.append
    say(f"photo: {inp.photo}  {W}x{H}  sha256={inp.sha256[:12]}...")
    say(f"background corners: "
        f"{ {k: round(v) for k, v in background_uniformity(gray).items()} }")

    bbox = None
    try:
        bbox = card_component_bbox(gray)
        say(f"component bbox: {bbox}")
    except RuntimeError as e:
        say(f"component segmentation FAILED: {e}")

    game = LORCANA
    coarse, ppm0 = coarse_locate(gray, game.card_w_mm, game.card_h_mm)
    say(f"coarse ppm: {ppm0}")
    for s in SIDES:
        c = coarse[s]
        say(f"coarse {s:6s}: {c.status:8s} pos={c.pos and round(c.pos, 1)} "
            f"n_ok={c.n_ok} mad={c.mad_px:.1f} method={c.method} {c.reason}")

    if ppm0 is None and bbox is not None:
        ppm0 = (bbox[2] - bbox[0]) / game.card_w_mm
    ppm_txt = f"{ppm0:.2f}" if ppm0 else "unknown"
    say(f"working px/mm: {ppm_txt}")

    rows_csv = [("stage", "scanner", "side", "u_px", "outcome", "v_px")]

    n_lines = 15
    rows = np.linspace(0.22 * H, 0.78 * H, n_lines)
    cols = np.linspace(0.22 * W, 0.78 * W, n_lines)
    cfg = {
        "left":   (W * 0.30, W * 0.30, W * 0.25, rows),
        "right":  (W * 0.70, W * 0.30, W * 0.25, rows),
        "top":    (H * 0.30, H * 0.30, H * 0.25, cols),
        "bottom": (H * 0.70, H * 0.30, H * 0.25, cols),
    }
    say("")
    say("== coarse per-line ==")
    for side, (approx, so, si, us) in cfg.items():
        for name, sc, kw in (
                ("texture", E.texture_scan, {"sustain": 30, "min_sep": 4.0}),
                ("step", E.step_scan, {"band": 5, "min_contrast": 25.0})):
            res = per_line(sc, gray, side, approx, us, so, si, **kw)
            oks = [v for (_, v, w) in res if w == "ok"]
            why = {}
            for (_, _, w) in res:
                if w != "ok":
                    why[w] = why.get(w, 0) + 1
            say(f"{side:6s} {name:7s}: {len(oks)}/{len(res)} ok  "
                + ", ".join(f"{n}x {k}" for k, n in sorted(why.items())))
            if oks:
                say(fmt_clusters(oks, ppm0))
            for (u, v, w) in res:
                rows_csv.append(("coarse", name, side, f"{u:.1f}", w,
                                 "" if v is None else f"{v:.2f}"))

    approx_pos = {}
    if args.bbox:
        bx = [float(v) for v in args.bbox.split(",")]
        assert len(bx) == 4, "--bbox needs x0,y0,x1,y1"
        approx_pos = {"left": bx[0], "right": bx[2],
                      "top": bx[1], "bottom": bx[3]}
        ppm0 = (bx[2] - bx[0]) / game.card_w_mm
        say(f"manual bbox {bx} -> ppm {ppm0:.2f}")
    for s in SIDES:
        if s in approx_pos:
            pass
        elif coarse[s].pos is not None:
            approx_pos[s] = float(coarse[s].pos)
        elif bbox is not None:
            approx_pos[s] = float({"left": bbox[0], "right": bbox[2],
                                   "top": bbox[1], "bottom": bbox[3]}[s])
    fine = {}
    if ppm0:
        ys = approx_pos.get("top", 0.22 * H), approx_pos.get("bottom", 0.78 * H)
        xs = approx_pos.get("left", 0.22 * W), approx_pos.get("right", 0.78 * W)
        frows = np.linspace(ys[0] + 0.15 * (ys[1] - ys[0]),
                            ys[0] + 0.85 * (ys[1] - ys[0]), args.n)
        fcols = np.linspace(xs[0] + 0.15 * (xs[1] - xs[0]),
                            xs[0] + 0.85 * (xs[1] - xs[0]), args.n)
        say("")
        say("== fine per-line (back.py params) ==")
        for side in SIDES:
            if side not in approx_pos:
                say(f"{side:6s}: no approx position -- skipped")
                continue
            us = frows if side in ("left", "right") else fcols
            so, si = 6.0 * ppm0, 3.0 * ppm0
            for name, sc, kw in (
                    ("step", E.step_scan, {}),
                    ("texture", E.texture_scan, {})):
                res = per_line(sc, gray, side, approx_pos[side], us, so, si,
                               **kw)
                fine.setdefault(side, {})[name] = res
                oks = [v for (_, v, w) in res if w == "ok"]
                why = {}
                for (_, _, w) in res:
                    if w != "ok":
                        why[w] = why.get(w, 0) + 1
                say(f"{side:6s} {name:7s}: {len(oks)}/{len(res)} ok  "
                    + ", ".join(f"{n}x {k}" for k, n in sorted(why.items())))
                if oks:
                    say(fmt_clusters(oks, ppm0))
                for (u, v, w) in res:
                    rows_csv.append(("fine", name, side, f"{u:.1f}", w,
                                     "" if v is None else f"{v:.2f}"))

    from PIL import Image, ImageDraw
    im = Image.fromarray(rgb).convert("RGB")
    dr = ImageDraw.Draw(im)
    lw = max(2, int(min(W, H) / 800))

    def draw_line(side, u, v, why, approx, so, si, dashed):
        col = COLORS.get(why, FALLBACK)
        if side in ("left", "top"):
            lo, hi = approx - so, approx + si
        else:
            lo, hi = approx - si, approx + so
        lo = max(0.0, lo)
        hi = min(float(W if side in ("left", "right") else H), hi)
        if side in ("left", "right"):
            seg = ((lo, u), (hi, u))
        else:
            seg = ((u, lo), (u, hi))
        if dashed:
            (x0, y0), (x1, y1) = seg
            n = 24
            for k in range(0, n, 2):
                a = (x0 + (x1 - x0) * k / n, y0 + (y1 - y0) * k / n)
                b = (x0 + (x1 - x0) * (k + 1) / n,
                     y0 + (y1 - y0) * (k + 1) / n)
                dr.line([a, b], fill=col, width=lw)
        else:
            dr.line([seg[0], seg[1]], fill=col, width=lw)
        if v is not None:
            p = (v, u) if side in ("left", "right") else (u, v)
            r = 3 * lw
            dr.ellipse([p[0] - r, p[1] - r, p[0] + r, p[1] + r],
                       outline=(255, 255, 255), width=lw, fill=col)

    for side, (approx, so, si, us) in cfg.items():
        if side in fine:
            fa, fso, fsi = approx_pos[side], 6.0 * ppm0, 3.0 * ppm0
            for (u, v, w) in fine[side]["step"]:
                draw_line(side, u, v, w, fa, fso, fsi, dashed=False)
        else:
            res = per_line(E.step_scan, gray, side, approx, us, so, si,
                           band=5, min_contrast=25.0)
            for (u, v, w) in res:
                draw_line(side, u, v, w, approx, so, si, dashed=True)
    if bbox is not None:
        dr.rectangle(list(bbox), outline=(0, 160, 255), width=lw)

    legend = ["solid=fine step scan, dashed=coarse step scan, dot=detection",
              "green=ok red=insufficient_contrast orange=glare_band "
              "magenta=shadowed gray=no_crossing blue=non_monotonic "
              "cyan-box=component bbox"]
    im.save(out_dir / "overlay.png")

    report = "\n".join(rep) + "\n\nlegend:\n" + "\n".join(legend) + "\n"
    for name, data in (("report.txt", report),
                       ("scanlines.csv",
                        "\n".join(",".join(map(str, r)) for r in rows_csv)
                        + "\n")):
        with open(out_dir / name, "wb") as f:
            f.write(data.encode("utf-8"))
            f.flush()
            os.fsync(f.fileno())
    print(report)
    print(f"wrote {out_dir}/report.txt, scanlines.csv, overlay.png")


if __name__ == "__main__":
    main()
