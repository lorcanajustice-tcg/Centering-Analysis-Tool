#!/usr/bin/env python3
"""Append anchor-survey rows for card_db renders missing from anchors_percard.csv.

Run after `update_db.py` whenever a new set drops:

    python card_db/anchor_survey.py            # append missing cards
    python card_db/anchor_survey.py --dry-run  # show what would be appended
    python card_db/anchor_survey.py --validate-emb 100   # convention self-check

PROVENANCE / CONVENTIONS
------------------------
The original 2026-07 survey script (which produced the initial 3,211 rows and
anchors_report.json) was lost with a discarded sandbox. This replacement was
reverse-validated against the existing CSV on 2026-07-19:

* Footer-emblem matching reproduces the legacy convention EXACTLY
  (100/100 randomly sampled legacy rows reproduce emb_score/x/y to 4 dp):
    - std template: images/6-20.jpg    gray[1900:2015, 650:820]  (170x115)
    - enc template: images/6-4-C2.jpg  gray[1900:2015, 650:820]
    - search window gray[1850:2048, 550:950], cv2.TM_CCOEFF_NORMED,
      reported (x, y) = best-match TOP-LEFT in full-image coordinates,
      score rounded to 4 decimals.

* Border scans (left/right/top/band_top/top_px_v2) are a documented
  RECONSTRUCTION: threshold crossings of column/row median profiles at
  black_level+30 (per the method note in anchors_report.json "verdict").
  On a 40-card legacy sample this reproduces legacy values within +-1px on
  ~90% of cards; the lost original sampled specific columns and is not
  bit-reproducible. Report aggregates (mode / pct-within-2px) are
  insensitive to +-1px.

* black_level here = median of the outer 40px frame ring. The lost
  original used an undetermined estimator; values agree in scale but not
  digit-for-digit. The column is descriptive only (never used by the
  analyzer pipelines).

Appended rows only - legacy rows are NEVER rewritten. Writes are
flush+fsync (mounted-folder safety) and the whole file is rewritten with
LF endings.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np

BASE = Path(__file__).resolve().parent      # card_db/
CSV_FILE = BASE / "anchors_percard.csv"
INDEX_FILE = BASE / "index.json"
IMAGES_DIR = BASE / "images"

CANON_SHAPE = (2048, 1468)                  # (h, w) of every standard render
STD_TPL_CARD = "images/6-20.jpg"            # legacy std-emblem template source
ENC_TPL_CARD = "images/6-4-C2.jpg"          # legacy enc-emblem template source
TPL_SLICE = (slice(1900, 2015), slice(650, 820))    # 170x115 template
WIN_SLICE = (slice(1850, 2048), slice(550, 950))    # emblem search window
WIN_X0, WIN_Y0 = 550, 1850

COLUMNS = ["id", "file", "setCode", "number", "promoGrouping", "rarity",
           "bordered_empirical", "black_level", "left_px", "right_px",
           "top_px", "band_top_px", "emb_score_std", "emb_x_std", "emb_y_std",
           "emb_score_enc", "emb_x_enc", "emb_y_enc", "top_px_v2"]


def _fsync_write(path: Path, data: bytes):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_templates():
    tpls = {}
    for tag, rel in (("std", STD_TPL_CARD), ("enc", ENC_TPL_CARD)):
        g = cv2.imread(str(BASE / rel), cv2.IMREAD_GRAYSCALE)
        if g is None:
            sys.exit(f"template source {rel} missing - run update_db.py first")
        tpls[tag] = g[TPL_SLICE]
    return tpls


def emblem_match(gray, tpl):
    """Legacy-convention footer-emblem match: (score_4dp, x, y) top-left."""
    win = gray[WIN_SLICE]
    res = cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED)
    _, mx, _, loc = cv2.minMaxLoc(res)
    return round(float(mx), 4), loc[0] + WIN_X0, loc[1] + WIN_Y0


def survey_one(gray, tpls):
    """Measure one render. Returns the row dict (id/file/etc filled by caller)."""
    ring = np.concatenate([gray[:, :40].ravel(), gray[:, -40:].ravel(),
                           gray[:40, :].ravel(), gray[-40:, :].ravel()])
    black = float(np.median(ring))

    colmed = np.median(gray, axis=0)
    rowmed = np.median(gray, axis=1)
    thr = black + 30
    left = int(np.argmax(colmed > thr))
    right = int(np.argmax(colmed[::-1] > thr))
    top = int(np.argmax(rowmed > thr))
    band = 1800 + int(np.argmax(rowmed[1800:] <= thr))
    # robust top variant (fraction-of-bright-pixels profile)
    rowfrac = (gray > thr).mean(axis=1)
    top_v2 = int(np.argmax(rowfrac > 0.5))

    # bordered iff a dark, plausibly-narrow frame exists on both sides
    bordered = (black < 10 and 40 <= left <= 200 and 40 <= right <= 200)

    row = {"black_level": black, "bordered_empirical": bordered}
    if bordered:
        row.update(left_px=left, right_px=right, top_px=top,
                   band_top_px=band, top_px_v2=top_v2)
    else:
        row.update(left_px=-1, right_px=-1, top_px=-1,
                   band_top_px=-1, top_px_v2=-1)
    for tag in ("std", "enc"):
        s, x, y = emblem_match(gray, tpls[tag])
        row[f"emb_score_{tag}"] = s
        row[f"emb_x_{tag}"] = x
        row[f"emb_y_{tag}"] = y
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--validate-emb", type=int, metavar="N", default=0,
                    help="re-measure N random legacy rows; report emb_* agreement")
    args = ap.parse_args()

    tpls = load_templates()
    raw = CSV_FILE.read_bytes().replace(b"\r\n", b"\n")
    existing = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
    have = {r["id"] for r in existing}

    if args.validate_emb:
        random.seed()
        ok = bad = skipped = 0
        for r in random.sample(existing, args.validate_emb):
            g = cv2.imread(str(BASE / r["file"]), cv2.IMREAD_GRAYSCALE)
            if g is None or g.shape != CANON_SHAPE:
                skipped += 1
                continue
            good = True
            for tag in ("std", "enc"):
                s, x, y = emblem_match(g, tpls[tag])
                if (abs(s - float(r[f"emb_score_{tag}"])) > 5e-5
                        or x != int(r[f"emb_x_{tag}"])
                        or y != int(r[f"emb_y_{tag}"])):
                    good = False
                    print(f"MISMATCH {r['id']} {tag}: csv "
                          f"{r[f'emb_score_{tag}']}@({r[f'emb_x_{tag}']},"
                          f"{r[f'emb_y_{tag}']}) got {s}@({x},{y})")
            ok += good
            bad += not good
        print(f"emb validation: {ok} exact, {bad} mismatched, {skipped} skipped")
        return

    index = json.loads(INDEX_FILE.read_text(encoding="utf-8"))["cards"]
    missing = [c for c in index if c["id"] not in have]
    print(f"{len(existing)} rows in CSV, {len(index)} cards in index, "
          f"{len(missing)} to survey")

    new_rows = []
    for c in missing:
        img = BASE / c["file"]
        g = cv2.imread(str(img), cv2.IMREAD_GRAYSCALE)
        if g is None:
            print(f"  SKIP {c['id']}: image missing/unreadable ({c['file']})")
            continue
        if g.shape != CANON_SHAPE:
            print(f"  SKIP {c['id']}: non-canonical size {g.shape} - "
                  f"survey conventions assume 1468x2048")
            continue
        row = survey_one(g, tpls)
        row.update(id=c["id"], file=c["file"], setCode=c["setCode"],
                   number=c["number"], promoGrouping=c["promoGrouping"] or "",
                   rarity=c["rarity"])
        new_rows.append(row)
        print(f"  {row['id']}: bordered={row['bordered_empirical']} "
              f"black={row['black_level']} "
              f"std {row['emb_score_std']}@({row['emb_x_std']},{row['emb_y_std']}) "
              f"enc {row['emb_score_enc']}@({row['emb_x_enc']},{row['emb_y_enc']})")

    if args.dry_run or not new_rows:
        print("dry-run - nothing written" if args.dry_run else "nothing to append")
        return

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator="\n")
    for r in new_rows:
        w.writerow(r)
    out = raw
    if not out.endswith(b"\n"):
        out += b"\n"
    out += buf.getvalue().encode("utf-8")
    _fsync_write(CSV_FILE, out)

    check = list(csv.DictReader(io.StringIO(
        CSV_FILE.read_bytes().decode("utf-8"))))
    assert len(check) == len(existing) + len(new_rows), "row count mismatch after write"
    print(f"appended {len(new_rows)} rows -> {len(check)} total")


if __name__ == "__main__":
    main()
