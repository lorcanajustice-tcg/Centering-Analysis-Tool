#!/usr/bin/env python3
"""Compare analyzer output against TAG grading-report reference numbers.

TAG (Technical Authentication & Grading) publishes a per-card scan of each
face, and the exports are conventionally named with the graded centering
baked into the filename:

    T6453597F-47L53R44T56B.jpg
    ^^^^^^^^ ^ ^^ ^^ ^^ ^^
    serial   |  |  |  |  bottom share, %
             |  |  |  top share, %
             |  |  right share, %
             |  left share, %
             face: F(ront) or B(ack)

This is a DEVELOPMENT tool. The numbers are a third party's measurement of
the same physical card, useful as an independent reference while working on
the detectors - they are deliberately kept out of the analysis path, which
must never see them. Nothing in `src/centering` imports this module.

Usage:

    python calibration/tag_reference.py <folder> [--game lorcana]
                                        [--card 13-244] [--json out.json]

Fronts need a card id for the render match; pass `--card` when every image
in the folder is the same card, otherwise the id is auto-detected per file.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

_PATTERN = re.compile(
    r"^(?P<serial>.*?)(?P<face>[FB])-"
    r"(?P<left>\d{1,3})L(?P<right>\d{1,3})R"
    r"(?P<top>\d{1,3})T(?P<bottom>\d{1,3})B$",
    re.IGNORECASE)


@dataclass
class TagReference:
    """Centering percentages parsed from a TAG export filename."""
    serial: str
    face: str          # "front" | "back"
    left: int
    right: int
    top: int
    bottom: int

    @property
    def lr_pct(self) -> float:
        return float(self.left)

    @property
    def tb_pct(self) -> float:
        return float(self.top)

    def __str__(self) -> str:
        return (f"{self.serial} {self.face}: "
                f"{self.left}/{self.right} LR, {self.top}/{self.bottom} TB")


def parse_tag_filename(name: str | Path) -> Optional[TagReference]:
    """TagReference for a TAG-style filename, or None if it isn't one.

    Deliberately strict: the shares on each axis must sum to 100, which is
    what stops an ordinary filename that happens to contain digits and
    letters from being read as a measurement.
    """
    stem = Path(name).stem
    m = _PATTERN.match(stem)
    if not m:
        return None
    left, right = int(m["left"]), int(m["right"])
    top, bottom = int(m["top"]), int(m["bottom"])
    if left + right != 100 or top + bottom != 100:
        return None
    return TagReference(
        serial=m["serial"], face="front" if m["face"].upper() == "F" else "back",
        left=left, right=right, top=top, bottom=bottom)


def _ratios(result, face: str):
    """(lr_pct, tb_pct) from a BackResult / BorderlessResult; None where refused."""
    if face == "back":
        lr, tb = result.ratio_lr, result.ratio_tb
    else:
        lr, tb = result.equivalent_ratio_lr, result.equivalent_ratio_tb
    return (lr.first_pct if lr else None), (tb.first_pct if tb else None)


def compare_folder(folder: Path, game, card_id: Optional[str] = None,
                   out_dir: Optional[str] = None) -> list[dict]:
    """Analyze every TAG-named image in `folder` and diff against its filename."""
    from centering.back import analyze_back
    from centering.borderless import analyze_borderless

    rows = []
    for path in sorted(folder.iterdir()):
        ref = parse_tag_filename(path.name)
        if ref is None:
            continue
        row = {"file": path.name, "face": ref.face, "reference": asdict(ref)}
        try:
            if ref.face == "back":
                res = analyze_back(str(path), game, out_dir=out_dir)
            else:
                cid = card_id
                if cid is None:
                    from centering.cache import DiskCache
                    from centering.identify import detect_card_id
                    db = Path(__file__).resolve().parents[1] / "card_db"
                    images = db / "images"
                    cid = detect_card_id(
                        path, db / "index.json",
                        images if images.exists() else None,
                        db / "sig_index.json", cache=DiskCache()).card_id
                    if cid is None:
                        raise RuntimeError(
                            "card id not auto-detected; pass --card")
                res = analyze_borderless(str(path), cid, game, out_dir=out_dir)
            lr, tb = _ratios(res, ref.face)
            row["measured"] = {"lr_pct": lr, "tb_pct": tb}
            row["delta_pts"] = {
                "lr": None if lr is None else round(lr - ref.lr_pct, 2),
                "tb": None if tb is None else round(tb - ref.tb_pct, 2)}
            row["px_per_mm"] = res.input.px_per_mm
            row["qa"] = [f.code for f in res.qa]
        except Exception as err:                      # noqa: BLE001 - report it
            row["error"] = f"{type(err).__name__}: {err}"
        rows.append(row)
    return rows


def _fmt(v, width=6):
    return " " * width if v is None else f"{v:>{width}.1f}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path, help="folder of TAG-named images")
    ap.add_argument("--game", default="lorcana")
    ap.add_argument("--card", default=None,
                    help="card id for front render matching (else auto-detected)")
    ap.add_argument("--out", default=None, help="write verification overlays here")
    ap.add_argument("--json", default=None, help="write the comparison as JSON")
    args = ap.parse_args(argv)

    if args.game != "lorcana":
        ap.error(f"unknown game {args.game!r}")
    from centering.games.lorcana import LORCANA

    rows = compare_folder(args.folder, LORCANA, card_id=args.card,
                          out_dir=args.out)
    if not rows:
        print(f"no TAG-named images found in {args.folder}")
        return 1

    print(f"{'file':<34} {'face':<6} {'TAG L/R':>8} {'ours':>6} {'d':>6} "
          f"{'TAG T/B':>8} {'ours':>6} {'d':>6}")
    for r in rows:
        ref = r["reference"]
        if "error" in r:
            print(f"{r['file']:<34} {r['face']:<6} {r['error']}")
            continue
        m, d = r["measured"], r["delta_pts"]
        print(f"{r['file']:<34} {r['face']:<6} "
              f"{ref['left']:>4}/{ref['right']:<3} {_fmt(m['lr_pct'])} {_fmt(d['lr'])} "
              f"{ref['top']:>4}/{ref['bottom']:<3} {_fmt(m['tb_pct'])} {_fmt(d['tb'])}")
        if r["qa"]:
            print(f"{'':<34} qa: {', '.join(r['qa'])}")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
