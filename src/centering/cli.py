"""Command-line version of the card centering checker.

Everything it prints is meant to be read by a person, not parsed by a
program. If you want the full numbers, use --json and read the file.
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

from .games.lorcana import LORCANA

GAMES = {"lorcana": LORCANA}


def _wrap(text, indent="    "):
    return textwrap.fill(text, width=78, initial_indent=indent,
                         subsequent_indent=indent)


def _give_or_take(unc, unit="mm"):
    return f"give or take {unc:.2f}{unit}" if unc is not None else ""


def _summary_back(r):
    lines = [f"BACK OF THE CARD  ({r.input.photo})",
             "  How far the printed gold frame sits from each cut edge."]
    for name, rr in (("Left to right", r.ratio_lr), ("Top to bottom", r.ratio_tb)):
        if rr.status in ("measured", "estimated"):
            tag = " (estimated)" if rr.status == "estimated" else ""
            lines.append(f"  {name}: {rr.display}{tag}   "
                         f"(exactly {rr.first_pct:.1f}%, "
                         f"give or take {rr.uncertainty_pts.total:.1f})")
        else:
            lines.append(f"  {name}: not measured")
            lines.append(_wrap(rr.refusal_reason or "", "      "))
            if rr.refusal_advice:
                lines.append(_wrap("What to do: " + rr.refusal_advice, "      "))
    names = {"left": "Left border  ", "right": "Right border ",
             "top": "Top border   ", "bottom": "Bottom border"}
    for s, m in r.borders_mm.items():
        if m.status in ("measured", "estimated"):
            pre = "~" if m.status == "estimated" else " "
            lines.append(f"  {names.get(s, s)}:{pre}{m.value:.2f}mm  "
                         f"{_give_or_take(m.uncertainty.total)}")
        else:
            lines.append(f"  {names.get(s, s)}: not measured")
            lines.append(_wrap(m.refusal_reason or "", "      "))
    return lines


def _summary_front(r):
    lines = [f"FRONT OF THE CARD  ({r.input.photo})",
             "  The artwork runs to the edge, so this measures how far the",
             "  printing sits off centre on the card."]
    for ax, (pos, neg) in (("x", ("to the right", "to the left")),
                           ("y", ("down", "up"))):
        m = r.shift_mm.get(ax)
        if m is None:
            continue
        if m.status in ("measured", "estimated"):
            direction = pos if m.value >= 0 else neg
            tag = "  (estimated, not a clean measurement)" \
                if m.status == "estimated" else ""
            lines.append(f"  {abs(m.value):.2f}mm too far {direction}  "
                         f"({_give_or_take(m.uncertainty.total)}){tag}")
        else:
            lines.append("  Not measured.")
            lines.append(_wrap(m.refusal_reason or "", "      "))
            if m.refusal_advice:
                lines.append(_wrap("What to do: " + m.refusal_advice, "      "))
    for name, rr in (("Left to right", r.equivalent_ratio_lr),
                     ("Top to bottom", r.equivalent_ratio_tb)):
        if rr and rr.status in ("measured", "estimated"):
            tag = " (estimated)" if rr.status == "estimated" else ""
            lines.append(f"  {name}, written the way graders write it: "
                         f"{rr.display}{tag}")
    if r.render:
        lines.append(f"  Matched to the official picture of the card using "
                     f"{r.render.n_inliers} points,")
        lines.append(f"  lining up to {r.render.median_reproj_px:.1f} pixels.")
    hx = r.hex_check
    if hx is not None and hx.status == "measured":
        if r.method == "ink_hexagon":
            lines.append(_wrap(
                "Measured from where the ink-cost hexagon sits: "
                f"{hx.centre_from_left_mm:.2f}mm from the left edge and "
                f"{hx.centre_from_top_mm:.2f}mm from the top, against "
                f"{hx.expected_from_left_mm:.2f}mm and "
                f"{hx.expected_from_top_mm:.2f}mm on a centred "
                f"{hx.layout} card.", "  "))
        elif hx.agreement_mm:
            parts = [f"{abs(v):.2f}mm {'across' if k == 'x' else 'down'}"
                     for k, v in hx.agreement_mm.items()]
            lines.append(_wrap(
                "Cross-check from the ink-cost hexagon agrees to within "
                + " and ".join(parts) + ".", "  "))
    return lines


def _emit(result, args, extra_lines=()):
    d = result.to_dict()
    if args.json:
        p = Path(args.json)
        p.write_text(json.dumps(d, indent=2))
    for ln in extra_lines:
        print(ln)
    qa = d.get("qa", [])
    for face_key in ("back", "front"):
        face = d.get(face_key)
        if isinstance(face, dict):
            qa = qa + face.get("qa", [])
    if qa:
        print("")
        print("THINGS WORTH KNOWING ABOUT THIS PHOTO")
        for q in qa:
            print(f"  * {q.get('title') or ''}")
            print(_wrap(q.get("message", ""), "    "))
            if q.get("advice"):
                print(_wrap("What to do: " + q["advice"], "    "))
    if args.json:
        print("")
        print(f"Every number from this check was saved to: {args.json}")


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="centering",
        description="Check how well a trading card is centred, from a photo.")
    ap.add_argument("--game", default="lorcana", choices=sorted(GAMES),
                    help="which game's cards these are")
    ap.add_argument("--out", default=None,
                    help="folder to save the marked-up pictures in")
    ap.add_argument("--json", default=None,
                    help="file to save every number to")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("back", help="check the back of a card")
    b.add_argument("photo", help="photo of the back")
    f = sub.add_parser("front", help="check the front of a card")
    f.add_argument("photo", help="photo of the front")
    f.add_argument("--card", default=None,
                   help='which card it is, e.g. "8-210", or part of its '
                        'name. Without it, the print shift is measured '
                        'from the ink-cost hexagon instead')
    k = sub.add_parser("corner",
                       help="check the front from a close-up of its "
                            "top-left corner")
    k.add_argument("photo", help="close-up of the top-left corner, with "
                                 "the whole ink-cost hexagon and some "
                                 "background beyond both edges")
    k.add_argument("--card", default=None,
                   help='which card it is, if known, e.g. "8-210"')
    c = sub.add_parser("card", help="check both sides and compare them")
    c.add_argument("--back", dest="back_photo", help="photo of the back")
    c.add_argument("--front", dest="front_photo", help="photo of the front")
    c.add_argument("--card", dest="card_id",
                   help='which card it is, e.g. "8-210", or part of its name')
    args = ap.parse_args(argv)
    game = GAMES[args.game]

    if args.cmd == "back":
        from .back import analyze_back
        r = analyze_back(args.photo, game, out_dir=args.out)
        _emit(r, args, _summary_back(r))
        if r.overlay:
            print(f"Marked-up picture, so you can check it yourself: {r.overlay}")
    elif args.cmd == "front":
        from .borderless import analyze_borderless
        r = analyze_borderless(args.photo, args.card, game, out_dir=args.out)
        _emit(r, args, _summary_front(r))
        if r.overlay:
            print(f"Marked-up picture, so you can check it yourself: {r.overlay}")
    elif args.cmd == "corner":
        from .hexanchor import analyze_corner
        r = analyze_corner(args.photo, game, card_id=args.card,
                           out_dir=args.out)
        lines = _summary_front(r)
        lines[0] = f"FRONT OF THE CARD, TOP-LEFT CORNER  ({r.input.photo})"
        _emit(r, args, lines)
        if r.overlay:
            print(f"Marked-up picture, so you can check it yourself: {r.overlay}")
    else:
        from .card_report import analyze_card
        r = analyze_card(back_photo=args.back_photo,
                         front_photo=args.front_photo,
                         card_id=args.card_id, game=game, out_dir=args.out)
        lines = []
        if r.back:
            lines += _summary_back(r.back)
        if r.front:
            lines += [""] + _summary_front(r.front)
        if r.registration_mm:
            lines += ["", "FRONT AND BACK CROSS-CHECK",
                      "  Both photos see the same card, cut once. A small",
                      "  disagreement is normal - the two sides are printed",
                      "  separately."]
            for ax, label in (("x", "Side to side"), ("y", "Up and down")):
                m = r.registration_mm.get(ax)
                if m is None:
                    continue
                if m.status == "measured":
                    lines.append(f"  {label}: they agree to "
                                 f"{abs(m.value):.2f}mm  "
                                 f"({_give_or_take(m.uncertainty.total)})")
                else:
                    lines.append(f"  {label}: could not be checked.")
                    lines.append(_wrap(m.refusal_reason or "", "      "))
        if r.mirror_consistency:
            lines += ["", _wrap(r.mirror_consistency, "  ")]
        _emit(r, args, lines)
        for face in (r.back, r.front):
            if face and face.overlay:
                print(f"Marked-up picture, so you can check it yourself: "
                      f"{face.overlay}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
