#!/usr/bin/env python3
"""Colour-segmentation re-measurement of the footer ink emblem.

WHY THIS EXISTS
---------------
`anchor_survey.py` locates the footer emblem by grayscale template
matching against two hand-picked templates (a standard-layout card and an
Enchanted one).  The ink glyph is NOT the same shape on every card - it is
the ink symbol, which differs per ink and is drawn differently again on
Enchanted / promo layouts - so a single template both (a) scores badly on
most cards, which is why only 147 of 375 full-art renders reached the
"high confidence" cut, and (b) drags the reported position toward whatever
part of the glyph happens to resemble the template.  That is the "template
bias" that made the anchor survey's x evidence self-contradictory
(full-art emblem x median 642 sd 8.6 on the std template vs 650 sd 6.3 on
the Enchanted one, against bordered 647 sd 2.2).

WHAT THIS MEASURES INSTEAD
--------------------------
The emblem's MIRROR-SYMMETRY AXIS in x, which is shape-agnostic:

1. Segment the emblem from the footer band by COLOUR distance (CIE Lab)
   from a local background model built from the same rows to the left and
   right of the emblem.  This works for every polarity we see in the
   corpus - bright glyph on the black band, rainbow hex on black, and the
   promo layouts' dark badge sitting on light artwork.
2. Take the connected component nearest the window centre.
3. Cross-correlate that component's mask with its own x-mirror.  The peak
   lag gives the axis of symmetry to sub-pixel precision; the peak height
   is a symmetry score that gates out the genuinely asymmetric glyphs
   (the Enchanted "fan") instead of silently biasing them.

The axis is a real print landmark: the ink symbol is centred on the card's
vertical centreline, so its axis in render pixels is a direct probe of the
horizontal render crop.  Nothing here uses a template, so bordered and
full-art renders are measured by exactly the same rule.

y is reported for the emblem too (component centroid + top/bottom) but is
NOT comparable across layouts: the two layouts do not even use the same
glyph (bordered cards carry the plain ink symbol, Epic/Enchanted full-art
cards carry the rainbow hex), and different glyphs have different vertical
extents.  That is also, in hindsight, why the template survey's bordered-vs
-full-art emblem y difference (+0.72 +- 0.15px) was never a crop
measurement - it was a glyph-shape difference.

THE SECOND ANCHOR: THE COLLECTOR LINE
-------------------------------------
For y we therefore use a different landmark, the bottom-left collector line
("20/204 - EN - 6").  It is the SAME text asset, same face and same size, on
both layouts, so its cap-top and baseline are directly comparable.  Measured
as the 50% crossings of a row profile of |pixel - row background|, which
picks up the glyphs whether they sit bright-on-black (standard footer) or
white-outlined on artwork (promo layouts).  Digits and "/" are present on
every card and share a cap height and a baseline, so the band edges do not
depend on which digits a card happens to carry.

CLI
---
    python card_db/emblem_colour.py --sample 40 --overlays   # eyeball it
    python card_db/emblem_colour.py                          # full survey
    python card_db/emblem_colour.py --report                 # stats only

Writes card_db/emblem_colour.csv (one row per render).  Never touches
anchors_percard.csv.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import collections
import random
import sys
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent
CSV_IN = BASE / "anchors_percard.csv"
INDEX_FILE = BASE / "index.json"
OUT_CSV = BASE / "emblem_colour.csv"

CANON_SHAPE = (2048, 1468)          # (h, w) of every standard render
WIN_X0, WIN_X1 = 554, 914           # 360 wide, centred on 734 = w/2
WIN_Y0, WIN_Y1 = 1856, 2032         # 176 tall, covers the footer band
CORE_HALF = 70                      # +-70px of window centre = "the emblem"
BG_MARGIN = 40                      # outer columns used for the bg model
MIN_AREA = 400                      # px; smaller components are noise
MAX_AREA = 40000                    # px; larger means we segmented the band
LAG_HALF = 60                       # +-60px of axis search (0.5px in lag)

# Collector-line ("20/204 - EN - 6") window and gates.
TXT_Y0, TXT_Y1, TXT_X0, TXT_X1 = 1962, 2032, 52, 336
TXT_CONTRAST = 40                   # gray levels above the row background
TXT_H_MIN, TXT_H_MAX = 30.0, 35.0   # px; the band is 32.4 on the standard layout

COLUMNS = ["id", "file", "setCode", "number", "promoGrouping", "rarity",
           "bordered_empirical", "ok", "reason", "axis_x", "sym_score",
           "area", "cx", "cy", "top_y", "bot_y", "left_x", "right_x",
           "bbox_cx", "polarity",
           "txt_ok", "txt_reason", "txt_top", "txt_bot", "txt_mid"]


# ---------------------------------------------------------------- segmentation

def _segment(bgr):
    """Return (mask, dist, reason) for the footer emblem in window coords."""
    win = bgr[WIN_Y0:WIN_Y1, WIN_X0:WIN_X1]
    lab = cv2.cvtColor(cv2.GaussianBlur(win, (3, 3), 0), cv2.COLOR_BGR2LAB)
    lab = lab.astype(np.float32)
    h, w = lab.shape[:2]

    # Background model: per-row median colour of the outer columns only, so a
    # vertical gradient in the artwork (or the band/art boundary) is followed
    # rather than smeared into the threshold.
    side = np.concatenate([lab[:, :BG_MARGIN], lab[:, -BG_MARGIN:]], axis=1)
    bg = np.median(side, axis=1)                       # (h, 3)
    dist = np.linalg.norm(lab - bg[:, None, :], axis=2)

    # Threshold from the background's own spread, floored so a perfectly flat
    # black band cannot drive the threshold to zero.
    spread = np.median(np.abs(side - bg[:, None, :]).sum(axis=2)) / 3.0
    thr = max(12.0, float(np.median(dist) + 4.0 * spread))
    mask = (dist > thr).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    n, lbl, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    if n <= 1:
        return None, dist, "no-component"

    # Pick the component whose centroid is closest to the window centreline and
    # that overlaps the core box; the emblem is always centred there.
    cx0, cy0 = w / 2.0, h / 2.0
    best, best_cost = None, None
    for i in range(1, n):
        a = stats[i, cv2.CC_STAT_AREA]
        if a < MIN_AREA or a > MAX_AREA:
            continue
        x, y = cents[i]
        if abs(x - cx0) > CORE_HALF:
            continue
        cost = abs(x - cx0) + 0.5 * abs(y - cy0) - 0.02 * a
        if best_cost is None or cost < best_cost:
            best, best_cost = i, cost
    if best is None:
        return None, dist, "no-core-component"

    comp = (lbl == best).astype(np.float32)
    # Fill interior holes: the glyph's own cut-outs are shape, not background,
    # and filling makes the mask depend on the outline alone.
    ff = comp.astype(np.uint8).copy()
    pad = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(ff, pad, (0, 0), 1)
    comp = np.maximum(comp, (ff == 0).astype(np.float32))
    return comp, dist, ""


def _symmetry_axis(comp):
    """Sub-pixel x of the mask's mirror-symmetry axis, plus a 0..1 score."""
    h, w = comp.shape
    cols = comp.sum(axis=0)
    nz = np.nonzero(cols)[0]
    if nz.size < 8:
        return None, 0.0
    x0, x1 = max(0, nz[0] - 8), min(w, nz[-1] + 9)
    sub = comp[:, x0:x1]
    flip = sub[:, ::-1]
    sw = sub.shape[1]

    energy = float((sub * sub).sum())
    if energy <= 0:
        return None, 0.0

    lags = range(-min(LAG_HALF, sw - 1), min(LAG_HALF, sw - 1) + 1)
    scores = []
    for k in lags:
        if k >= 0:
            a, b = sub[:, k:], flip[:, :sw - k]
        else:
            a, b = sub[:, :sw + k], flip[:, -k:]
        ov = min(a.shape[1], b.shape[1])
        if ov < sw // 2:
            scores.append(0.0)
            continue
        scores.append(float((a[:, :ov] * b[:, :ov]).sum()) / energy)
    scores = np.array(scores)
    j = int(np.argmax(scores))
    if j == 0 or j == len(scores) - 1:
        return None, float(scores[j])

    # Parabolic sub-lag refinement.
    y0, y1, y2 = scores[j - 1], scores[j], scores[j + 1]
    denom = (y0 - 2 * y1 + y2)
    delta = 0.0 if denom == 0 else 0.5 * (y0 - y2) / denom
    k = list(lags)[j] + float(np.clip(delta, -1, 1))

    # Symmetry about axis a maps x -> 2a - x; correlating sub with its own
    # flip peaks at lag k = 2a - (sw - 1), in sub-window coordinates.
    axis_sub = (k + sw - 1) / 2.0
    return x0 + axis_sub, float(y1)


def collector_band(bgr):
    """Cap-top / baseline of the bottom-left collector line, sub-pixel.

    Returns (top, bot, reason).  The row profile is |pixel - row median| with
    a contrast floor, so it responds to dark-outlined white text on artwork as
    well as to plain white text on the black footer band.
    """
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    g = g[TXT_Y0:TXT_Y1, TXT_X0:TXT_X1]
    bg = np.median(g, axis=1)[:, None]
    prof = np.clip(np.abs(g - bg) - TXT_CONTRAST, 0, None).sum(axis=1)
    if prof.max() <= 0:
        return None, None, "no-text"
    half = 0.5 * prof.max()
    idx = np.nonzero(prof >= half)[0]
    if idx.size < 8:
        return None, None, "text-too-thin"
    a, b = int(idx[0]), int(idx[-1])
    if a == 0 or b == len(prof) - 1:
        return None, None, "text-clipped"
    top = a - 1 + (half - prof[a - 1]) / (prof[a] - prof[a - 1])
    bot = b + (prof[b] - half) / (prof[b] - prof[b + 1])
    h = bot - top
    if not (TXT_H_MIN <= h <= TXT_H_MAX):
        return None, None, f"text-height-{h:.1f}"
    return TXT_Y0 + float(top), TXT_Y0 + float(bot), ""


def measure(bgr):
    comp, dist, reason = _segment(bgr)
    if comp is None:
        return _add_collector(bgr, {"ok": False, "reason": reason})
    axis, score = _symmetry_axis(comp)
    ys, xs = np.nonzero(comp)
    area = int(comp.sum())
    row = {
        "ok": axis is not None,
        "reason": "" if axis is not None else "no-symmetry-peak",
        "axis_x": None if axis is None else round(WIN_X0 + axis, 3),
        "sym_score": round(score, 4),
        "area": area,
        "cx": round(WIN_X0 + float(xs.mean()), 3),
        "cy": round(WIN_Y0 + float(ys.mean()), 3),
        "top_y": WIN_Y0 + int(ys.min()),
        "bot_y": WIN_Y0 + int(ys.max()),
        "left_x": WIN_X0 + int(xs.min()),
        "right_x": WIN_X0 + int(xs.max()),
    }
    row["bbox_cx"] = round((row["left_x"] + row["right_x"]) / 2.0, 3)
    # polarity: is the emblem brighter or darker than its local background?
    win = bgr[WIN_Y0:WIN_Y1, WIN_X0:WIN_X1]
    g = cv2.cvtColor(win, cv2.COLOR_BGR2GRAY).astype(np.float32)
    m = comp > 0
    side = np.concatenate([g[:, :BG_MARGIN], g[:, -BG_MARGIN:]], axis=1)
    row["polarity"] = "bright" if g[m].mean() > np.median(side) else "dark"
    _add_collector(bgr, row)
    return row


def _add_collector(bgr, row):
    t, b, why = collector_band(bgr)
    row["txt_ok"] = t is not None
    row["txt_reason"] = why
    row["txt_top"] = None if t is None else round(t, 4)
    row["txt_bot"] = None if b is None else round(b, 4)
    row["txt_mid"] = None if t is None else round((t + b) / 2.0, 4)
    return row


# ---------------------------------------------------------------------- report

SYM_MIN = 0.95              # mirror-symmetry score a glyph must reach
AXIS_WINDOW = 1.0           # px; reject axes this far from the population median
PPM_R = 1468 / 62.9         # render px per mm (1468px over the 62.9mm card)

# Systematic floor from --selftest: recovering an imposed 0.25px shift on real
# renders leaves a mean error of 0.07px (emblem axis) / 0.04px (collector line),
# i.e. the estimators carry a sub-pixel phase bias of that order.  Both
# populations are measured with the same estimator on the same pixel grid so
# most of it cancels, but the statistical sem alone would badly overstate what
# a 0.04px population difference is worth.  Bounds below are quoted against
# quadrature-summed statistical + systematic error.
SYS_FLOOR_PX = 0.10


def _robust(vals, k=3.0):
    """(n, mean, sd, sem) after a median/MAD 3-sigma trim."""
    a = np.asarray(vals, dtype=float)
    med = np.median(a)
    mad = 1.4826 * np.median(np.abs(a - med))
    keep = a[np.abs(a - med) <= max(k * mad, 0.05)]
    sd = keep.std(ddof=1) if keep.size > 1 else float("nan")
    return keep.size, keep.mean(), sd, sd / max(keep.size, 1) ** 0.5, med


def _compare(name, bvals, fvals, out):
    nb, mb, sb, eb, db = _robust(bvals)
    nf, mf, sf, ef, df = _robust(fvals)
    d = mf - mb
    se = (ef ** 2 + eb ** 2) ** 0.5
    tot = (se ** 2 + SYS_FLOOR_PX ** 2) ** 0.5
    bound = abs(d) + 1.96 * tot
    out.append(f"  {name}")
    out.append(f"    bordered  n={nb:5d}  mean={mb:10.4f}  sd={sb:7.4f}  sem={eb:7.4f}")
    out.append(f"    full-art  n={nf:5d}  mean={mf:10.4f}  sd={sf:7.4f}  sem={ef:7.4f}")
    out.append(f"    full-art - bordered = {d:+.4f} +- {se:.4f} (stat)"
               f" +- {SYS_FLOOR_PX:.2f} (sys) px"
               f"  = {d / PPM_R * 1000:+.2f} +- {tot / PPM_R * 1000:.2f} um")
    out.append(f"    95% bound on |offset|  = {bound:.3f} px"
               f" = {bound / PPM_R * 1000:.1f} um = {bound / PPM_R:.4f} mm")
    return d, se


def report(path):
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8")))
    out = [f"emblem_colour report over {len(rows)} renders", ""]

    emb = [r for r in rows if r["ok"] == "True"]
    fails = collections.Counter(r["reason"] for r in rows if r["ok"] != "True")
    out.append(f"emblem segmented on {len(emb)} / {len(rows)}; failures {dict(fails)}")
    for r in emb:
        r["_ax"] = float(r["axis_x"])
        r["_s"] = float(r["sym_score"])
    med = np.median([r["_ax"] for r in emb])
    keep = [r for r in emb if r["_s"] >= SYM_MIN and abs(r["_ax"] - med) <= AXIS_WINDOW]
    out.append(f"after sym_score >= {SYM_MIN} and |axis - median| <= {AXIS_WINDOW}px: "
               f"{len(keep)} rows")
    out.append("")
    out.append("ANCHOR 1 - emblem mirror-symmetry axis (x). Shape-agnostic, so it")
    out.append("compares across layouts even though the glyphs differ.")
    _compare("axis_x", [r["_ax"] for r in keep if r["bordered_empirical"] == "True"],
             [r["_ax"] for r in keep if r["bordered_empirical"] != "True"], out)

    txt = [r for r in rows if r["txt_ok"] == "True"]
    tfails = collections.Counter(r["txt_reason"] for r in rows if r["txt_ok"] != "True")
    out.append("")
    out.append(f"ANCHOR 2 - collector line (y). Same text asset on both layouts.")
    out.append(f"  measured on {len(txt)} / {len(rows)}; failures {dict(tfails)}")
    for key in ("txt_top", "txt_bot", "txt_mid"):
        _compare(key,
                 [float(r[key]) for r in txt if r["bordered_empirical"] == "True"],
                 [float(r[key]) for r in txt if r["bordered_empirical"] != "True"],
                 out)
    return "\n".join(out)


# -------------------------------------------------------------------- selftest

def selftest(rows, shifts=(0.25, 0.5, 1.0), n=12):
    """Shift real renders by a known sub-pixel amount and recover it.

    This is what licenses reading a 0.04px population difference as a
    measurement rather than as noise: if the estimators cannot recover an
    imposed 0.25px shift to better than that, the comparison means nothing.
    """
    lines = ["sub-pixel self-test (imposed shift -> recovered shift, px)", ""]
    for dx in shifts:
        ex, ey = [], []
        for r in rows[:n]:
            img = cv2.imread(str(BASE / r["file"]))
            if img is None or img.shape[:2] != CANON_SHAPE:
                continue
            base = measure(img)
            M = np.float32([[1, 0, dx], [0, 1, dx]])
            sh = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                                flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_REPLICATE)
            got = measure(sh)
            if base.get("ok") and got.get("ok"):
                ex.append(got["axis_x"] - base["axis_x"] - dx)
            if base.get("txt_ok") and got.get("txt_ok"):
                ey.append(got["txt_mid"] - base["txt_mid"] - dx)
        for nm, e in (("emblem axis_x", ex), ("collector txt_mid", ey)):
            a = np.asarray(e)
            lines.append(f"  shift {dx:4.2f}px  {nm:18s} n={a.size:3d} "
                         f"error mean={a.mean():+.4f} sd={a.std(ddof=1):.4f} "
                         f"max|e|={np.abs(a).max():.4f}")
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------- driver

def _rows():
    raw = CSV_IN.read_bytes().replace(b"\r\n", b"\n").decode("utf-8")
    return list(csv.DictReader(io.StringIO(raw)))


def _fsync_write(path: Path, data: bytes):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=0,
                    help="measure N random cards instead of the whole corpus")
    ap.add_argument("--only", default="", help="comma-separated card ids")
    ap.add_argument("--overlays", action="store_true",
                    help="write annotated footer crops to results/emblem_probe/")
    ap.add_argument("--out", default=str(OUT_CSV))
    ap.add_argument("--start", type=int, default=0,
                    help="index of the first card to measure (chunked runs)")
    ap.add_argument("--count", type=int, default=0,
                    help="how many cards to measure from --start (0 = all)")
    ap.add_argument("--no-header", action="store_true",
                    help="omit the CSV header (for chunk files that get concatenated)")
    ap.add_argument("--selftest", action="store_true",
                    help="recover imposed sub-pixel shifts on real renders")
    ap.add_argument("--report", action="store_true",
                    help="print the population comparison from an existing --out csv")
    args = ap.parse_args()

    if args.report:
        print(report(args.out))
        return

    if args.selftest:
        rows = [r for r in _rows() if r["bordered_empirical"] == "True"]
        random.seed(5)
        random.shuffle(rows)
        print(selftest(rows))
        return

    rows = _rows()
    if args.only:
        want = set(args.only.split(","))
        rows = [r for r in rows if r["id"] in want]
    elif args.sample:
        random.seed(11)
        rows = random.sample(rows, min(args.sample, len(rows)))

    if args.start or args.count:
        rows = rows[args.start:args.start + args.count if args.count else None]

    ovdir = BASE.parent / "results" / "emblem_probe"
    if args.overlays:
        ovdir.mkdir(parents=True, exist_ok=True)

    out, bad = [], 0
    for i, r in enumerate(rows):
        img = cv2.imread(str(BASE / r["file"]))
        if img is None or img.shape[:2] != CANON_SHAPE:
            rec = {"ok": False, "reason": "missing-or-noncanonical"}
        else:
            rec = measure(img)
        rec.update(id=r["id"], file=r["file"], setCode=r["setCode"],
                   number=r["number"], promoGrouping=r["promoGrouping"],
                   rarity=r["rarity"],
                   bordered_empirical=r["bordered_empirical"])
        out.append(rec)
        bad += not rec["ok"]
        if args.overlays and img is not None:
            crop = img[WIN_Y0:WIN_Y1, WIN_X0:WIN_X1].copy()
            if rec.get("axis_x"):
                ax = int(round(rec["axis_x"] - WIN_X0))
                cv2.line(crop, (ax, 0), (ax, crop.shape[0]), (0, 0, 255), 1)
                cv2.rectangle(crop,
                              (rec["left_x"] - WIN_X0, rec["top_y"] - WIN_Y0),
                              (rec["right_x"] - WIN_X0, rec["bot_y"] - WIN_Y0),
                              (0, 255, 0), 1)
            cv2.line(crop, (734 - WIN_X0, 0), (734 - WIN_X0, crop.shape[0]),
                     (255, 255, 0), 1)
            cv2.putText(crop, f"{r['id']} {rec.get('sym_score')}", (4, 14), 0,
                        0.45, (0, 0, 255), 1)
            cv2.imwrite(str(ovdir / f"ov_{r['id']}.png"), crop)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator="\n",
                       extrasaction="ignore")
    if not args.no_header:
        w.writeheader()
    for r in out:
        w.writerow(r)
    _fsync_write(Path(args.out), buf.getvalue().encode("utf-8"))
    print(f"measured {len(out)} renders, {bad} unmeasurable -> {args.out}")


if __name__ == "__main__":
    main()
