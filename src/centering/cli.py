"""Thin CLI over the centering library."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .games.lorcana import LORCANA

GAMES = {"lorcana": LORCANA}


def _summary_back(r):
    lines = [f"BACK  ({r.input.photo})"]
    rl, rt = r.ratio_lr, r.ratio_tb
    for name, rr in (("L/R", rl), ("T/B", rt)):
        if rr.status == "measured":
            lines.append(f"  {name}: {rr.display}  ({rr.first_pct:.1f} "
                         f"+- {rr.uncertainty_pts.total:.1f} pts)")
        else:
            lines.append(f"  {name}: refused - {rr.refusal_reason}")
    for s, m in r.borders_mm.items():
        if m.status == "measured":
            lines.append(f"  {s:>6}: {m.value:.3f} mm +- {m.uncertainty.total:.3f}")
    return lines


def _summary_front(r):
    lines = [f"FRONT ({r.input.photo})"]
    for ax, label in (("x", "horizontal"), ("y", "vertical")):
        m = r.shift_mm.get(ax)
        if m and m.status == "measured":
            direc = {("x", 1): "right", ("x", -1): "left",
                     ("y", 1): "bottom", ("y", -1): "top"}[
                         (ax, 1 if m.value >= 0 else -1)]
            lines.append(f"  print shift {label}: {abs(m.value):.3f} mm toward "
                         f"{direc} edge (+- {m.uncertainty.total:.3f})")
        elif m:
            lines.append(f"  shift {ax}: refused - {m.refusal_reason}")
    for name, rr in (("equiv L/R", r.equivalent_ratio_lr),
                     ("equiv T/B", r.equivalent_ratio_tb)):
        if rr and rr.status == "measured":
            lines.append(f"  {name}: {rr.display} (convention: nominal margin)")
    if r.render:
        lines.append(f"  render match: {r.render.n_inliers} inliers, "
                     f"{r.render.median_reproj_px:.2f}px reprojection")
    return lines


def _emit(result, args, extra_lines=()):
    d = result.to_dict()
    if args.json:
        p = Path(args.json)
        p.write_text(json.dumps(d, indent=2))
        print(f"json: {p}")
    for ln in extra_lines:
        print(ln)
    qa = d.get("qa", [])
    for face_key in ("back", "front"):
        face = d.get(face_key)
        if isinstance(face, dict):
            qa = qa + face.get("qa", [])
    if qa:
        print("QA flags:")
        for q in qa:
            print(f"  [{q['code']}] {q['message']}")
    tilt = d.get("tilt") or {}
    if not tilt:
        for face_key in ("back", "front"):
            if isinstance(d.get(face_key), dict) and d[face_key].get("tilt"):
                t = d[face_key]["tilt"]
                print(f"  {face_key} tilt: total "
                      f"{t.get('total_deg')} deg, keystone w/h "
                      f"{t.get('keystone_w_pct')}%/{t.get('keystone_h_pct')}%, "
                      f"corrected={t.get('corrected')}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="centering",
                                 description="Card centering analysis from phone photos")
    ap.add_argument("--game", default="lorcana", choices=sorted(GAMES))
    ap.add_argument("--out", default=None, help="output dir for overlays")
    ap.add_argument("--json", default=None, help="write full result JSON here")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("back", help="bordered face (frame-line) analysis")
    b.add_argument("photo")
    f = sub.add_parser("front", help="borderless face (render-match) analysis")
    f.add_argument("photo")
    f.add_argument("--card", required=True,
                   help='card id: "6/C2", "7:69", or a unique name substring')
    c = sub.add_parser("card", help="combined front+back report")
    c.add_argument("--back", dest="back_photo")
    c.add_argument("--front", dest="front_photo")
    c.add_argument("--card", dest="card_id")
    args = ap.parse_args(argv)
    game = GAMES[args.game]

    if args.cmd == "back":
        from .back import analyze_back
        r = analyze_back(args.photo, game, out_dir=args.out)
        _emit(r, args, _summary_back(r))
        if r.overlay:
            print(f"overlay: {r.overlay}")
    elif args.cmd == "front":
        from .borderless import analyze_borderless
        r = analyze_borderless(args.photo, args.card, game, out_dir=args.out)
        _emit(r, args, _summary_front(r))
        if r.overlay:
            print(f"overlay: {r.overlay}")
    else:
        from .card_report import analyze_card
        r = analyze_card(back_photo=args.back_photo,
                         front_photo=args.front_photo,
                         card_id=args.card_id, game=game, out_dir=args.out)
        lines = []
        if r.back:
            lines += _summary_back(r.back)
        if r.front:
            lines += _summary_front(r.front)
        if r.registration_mm:
            m = r.registration_mm.get("x")
            if m and m.status == "measured":
                lines.append(f"front-back registration x: {m.value:+.3f} mm "
                             f"(+- {m.uncertainty.total:.3f})")
            my = r.registration_mm.get("y")
            if my and my.status == "refused":
                lines.append(f"registration y: refused - {my.refusal_reason}")
        if r.mirror_consistency:
            lines.append(r.mirror_consistency)
        _emit(r, args, lines)
        for face in (r.back, r.front):
            if face and face.overlay:
                print(f"overlay: {face.overlay}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
