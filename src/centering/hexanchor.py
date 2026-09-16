"""Card-agnostic front check from the printed ink-cost hexagon.

Every Lorcana front prints the ink-cost hexagon at a fixed place in the
layout (survey of all 3,226 official pictures: the centre is fixed to
~0.005px within a layout). So where the hexagon sits against the LEFT and
TOP cut edges says how far the print is shifted, without knowing which
card it is and without the official picture.

Two ways in:

- `measure_in_card`: the four cut edges are already found (the normal
  front pipeline). The card is straightened, the hexagon is fitted, and
  the card's own size is the ruler. With a known card this is a
  cross-check on the official-picture result; without one it is the
  result.
- `analyze_corner`: a close-up of the top-left corner only. The hexagon
  itself is the ruler (its printed size is known), and only the left and
  top edges are needed.

Layout choice: a known card uses its own row of the survey. An unknown
card is assumed to be from a recent set (set 4 on): sets 1-3 printed the
uninkable hexagon 0.14mm further right, and that is said in a note.
"""
from __future__ import annotations

import csv
import math
from functools import lru_cache
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import edges as E
from . import geometry as G
from . import hexfit as HF
from .fitting import _colour_edge, _edge_report, _prefer_fit, _thin_fit
from .games.base import GameSpec, HexAnchorSpec
from .imgio import load_photo
from .types import (BorderlessResult, HexAnchorReport, Measurement, QAFlag,
                    Ratio, TiltReport, Uncertainty)

# a ring's two edges: outer/inner apothem ratio. Standard 1.153-1.160;
# the thicker-ring printing used on ~350 cards (mostly sets 5 and 7) 1.175
RING_RATIO = (1.12, 1.20)
# fitted hexagon size vs the layout it is matched to (full card)
LAYOUT_SIZE_TOL = 0.05
# smallest inner apothem worth fitting, px
MIN_APOTHEM_PX = 14.0
# close-up: a layout is only believable if it puts the print within this
# of centre in both directions. A wrong layout reads 1mm or more off in
# both (the hexagon sizes differ by 17%)
PLAUSIBLE_SHIFT_MM = 1.0
# when two designs both fit: print shifts are usually small (fixture fronts
# reach 0.52mm), so the design that puts the print nearer centre is
# favoured, with the flower-shaped surround as one more vote. A few
# uninkable promo frames carry an outline the flower test mistakes for it,
# which is why it is only a vote.
SHIFT_PRIOR_MM = 0.35
SWIRL_VOTE = 1.0
# close-up: the left and top edges must be roughly square to each other.
# The angle between two short edge fits is itself noisy (the same photo
# reads 89.2 or 90.0 degrees depending on the crop), and a skew that is the
# same over the corner does not move distances measured along the axes to
# the two edges; beyond this the photo is too oblique to trust.
CORNER_SQUARE_DEG = 2.0
# close-up edge search, mm outside / inside the expected edge
CORNER_SEARCH_OUT_MM = 2.5
CORNER_SEARCH_IN_MM = 1.2
# stay clear of the rounded corner, mm from the other edge
CORNER_CLEAR_MM = 4.0

RESHOOT_CORNER = (
    "Take the close-up square on, with the card flat on plain paper and a "
    "clear strip of background showing beyond the top and left edges, and "
    "the whole ink-cost hexagon sharply in focus.")


# --------------------------------------------------------------------------
# layouts
# --------------------------------------------------------------------------

def _mm_from_render(spec: HexAnchorSpec, game: GameSpec, centre_px):
    """Where a hexagon centre at `centre_px` (official picture) sits on a
    perfectly centred card: (from left, from top, offset x, offset y) mm.
    Offsets are from the print centre, which is the picture centre less
    the calibrated crop bias."""
    wr, hr = spec.render_size
    r = spec.render_px_per_mm
    bias = game.render_crop_bias_mm or {"x": 0.0, "y": 0.0}
    off_x = (centre_px[0] - wr / 2.0) / r
    off_y = (centre_px[1] - hr / 2.0) / r
    return (game.card_w_mm / 2.0 + bias["x"] + off_x,
            game.card_h_mm / 2.0 + bias["y"] + off_y, off_x, off_y)


@lru_cache(maxsize=4)
def _percard_rows(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open(newline="", encoding="utf-8") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def _card_key(card: dict) -> Optional[str]:
    try:
        key = f"{card['setCode']}-{int(card['number'])}"
    except (KeyError, TypeError, ValueError):
        return None
    promo = card.get("promoGrouping")
    return f"{key}-{promo}" if promo else key


def _layout_from_row(spec: HexAnchorSpec, row: dict):
    """(layout name, centre px) from a survey row, or None when this
    card's hexagon matches no known layout."""
    if not row.get("apothems"):
        return None
    ap = [float(v) for v in row["apothems"].split(";")]
    cxs = [float(v) for v in row["cx_each"].split(";")]
    cys = [float(v) for v in row["cy_each"].split(";")]
    for name, lay in spec.layouts.items():
        idx = []
        for ring in lay.rings:
            target = 0.5 * (ring[2] + ring[3])
            j = [i for i, a in enumerate(ap) if abs(a - target) < 1.2]
            if not j:
                break
            idx.append(j[0])
        else:
            return name, (float(np.mean([cxs[i] for i in idx])),
                          float(np.mean([cys[i] for i in idx])))
    return None


def candidate_layouts(spec: HexAnchorSpec, card: Optional[dict] = None,
                      db_dir=None):
    """Layouts to try: list of (name, centre_px, plain-words source).

    Raises ValueError (plain words) when the card is known and its
    hexagon is printed in a layout of its own."""
    if card:
        key = _card_key(card)
        if key and spec.percard_csv and db_dir:
            row = _percard_rows(str(Path(db_dir) / spec.percard_csv)).get(key)
            if row is not None:
                got = _layout_from_row(spec, row)
                if got is None:
                    raise ValueError(
                        "this card's ink-cost hexagon is printed in a "
                        "special layout of its own, so its position says "
                        "nothing about centring")
                return [(got[0], got[1],
                         "this card's own entry in the survey of every "
                         "official card picture")]
        setc = str(card.get("setCode", ""))
        out = []
        for name in spec.default_layouts:
            centre = spec.layouts[name].centre_px
            src = f"the usual {name} layout for set {setc}"
            for (lname, sets), c in spec.older_layouts.items():
                if lname == name and setc in sets:
                    centre = c
                    src = f"the older {name} layout used in sets " \
                          f"{', '.join(sets)}"
            out.append((name, centre, src))
        return out
    return [(name, spec.layouts[name].centre_px,
             f"the usual {name} layout (card not known, so a set from 4 "
             "on is assumed)") for name in spec.default_layouts]


def _older_layout_note(spec: HexAnchorSpec, layout: str, game: GameSpec):
    for (lname, sets), c in spec.older_layouts.items():
        if lname != layout:
            continue
        x_new = _mm_from_render(spec, game, spec.layouts[lname].centre_px)[0]
        x_old = _mm_from_render(spec, game, c)[0]
        d = x_old - x_new
        return (f"The card was not identified, so it is assumed to be from "
                f"set 4 or later. {layout.capitalize()} cards from sets "
                f"{', '.join(sets)} print the hexagon {abs(d):.2f}mm "
                f"further {'right' if d > 0 else 'left'}; for one of those, "
                f"the left-to-right shift is really {abs(d):.2f}mm further "
                f"{'left' if d > 0 else 'right'} than shown.")
    return None


# --------------------------------------------------------------------------
# picking the ring out of the fitted boundaries
# --------------------------------------------------------------------------

def _ring_pairs(bounds):
    """(inner, outer) boundary pairs whose size ratio fits a ring."""
    out = []
    for i in range(len(bounds)):
        for j in range(i + 1, len(bounds)):
            q = bounds[j].apothem / bounds[i].apothem
            if RING_RATIO[0] <= q <= RING_RATIO[1]:
                out.append((bounds[i], bounds[j]))
    return out


# the inkable surround's inner outline, as a multiple of the ring's inner
# edge: 1.248 standard, 1.253 thick-ring printing. A few uninkable promos
# carry an outline at 1.259, just outside.
SWIRL_RATIO = (1.235, 1.256)


def _swirl_seen(bounds, inner):
    return any(SWIRL_RATIO[0] <= b.apothem / inner.apothem <= SWIRL_RATIO[1]
               for b in bounds)


def _ring_constants(spec: HexAnchorSpec, layout: str):
    """Per ring edge: (vertical, slanted) apothem mm and centre offset mm."""
    lay = spec.layouts[layout]
    r = spec.render_px_per_mm
    return ([(ring[2] / r, ring[3] / r) for ring in lay.rings],
            [(ring[0] / r, ring[1] / r) for ring in lay.rings])


def _joint(spec, layout, pair):
    ap, offs = _ring_constants(spec, layout)
    return HF.joint_fit(list(pair), ap, offs)


# --------------------------------------------------------------------------
# edge-definition helpers
# --------------------------------------------------------------------------

def _line_stat_mm(line, ppm):
    if line is None or not getattr(line, "n", 0):
        return 0.0
    return line.rms / math.sqrt(max(line.n, 1)) / ppm


def _shift_measurement(value, stat, persp, terms):
    return Measurement(value, "mm", Uncertainty(
        statistical=stat, perspective=persp,
        edge_definition=math.sqrt(sum(t * t for t in terms))))


# --------------------------------------------------------------------------
# full card
# --------------------------------------------------------------------------

def measure_in_card(rgb, quad, game: GameSpec, px_per_mm: float,
                    lines: dict, methods: dict, est_extra: dict,
                    card: Optional[dict] = None,
                    db_dir=None) -> HexAnchorReport:
    """Hexagon check on a photo whose four cut edges are known.

    `quad` is the card's corners (TL, TR, BR, BL, photo px); `lines`,
    `methods`, `est_extra` are the front pipeline's per-edge fits,
    detector names and estimate-tier extra uncertainty (mm)."""
    spec = game.hex_anchor
    if spec is None:
        return HexAnchorReport.refused(
            "full_card", "there is no ink-cost hexagon layout on record for "
            "this game")
    try:
        cands = candidate_layouts(spec, card, db_dir)
    except ValueError as err:
        return HexAnchorReport.refused("full_card", str(err))
    if rgb is None or rgb.ndim != 3:
        return HexAnchorReport.refused(
            "full_card", "the check needs a colour photo")

    s = float(px_per_mm)
    ex = [_mm_from_render(spec, game, c) for _, c, _ in cands]
    gx = float(np.mean([e[0] for e in ex]))
    gy = float(np.mean([e[1] for e in ex]))
    a_min = min(spec.layouts[n].rings[0][2] for n, _, _ in cands)
    a_max = max(spec.layouts[n].rings[-1][3] for n, _, _ in cands)
    r = spec.render_px_per_mm
    if a_min / r * s < MIN_APOTHEM_PX:
        return HexAnchorReport.refused(
            "full_card", f"the photo is too small: the hexagon is only "
            f"{2 * a_min / r * s:.0f} pixels across")

    # straighten the top-left of the card at the photo's own resolution
    Hm = G.homography_to_card(quad, game.card_w_mm, game.card_h_mm)
    S = np.diag([s, s, 1.0]) @ Hm
    size_mm = max(gx, gy) + spec.search_mm + 1.3 * a_max / r + 1.0
    n = int(math.ceil(size_mm * s))
    patch = cv2.warpPerspective(np.ascontiguousarray(rgb), S, (n, n),
                                flags=cv2.INTER_CUBIC,
                                borderMode=cv2.BORDER_REPLICATE)
    bounds = HF.find_boundaries(
        patch, (gx * s, gy * s), spec.search_mm * s,
        (0.85 * a_min / r * s, 1.05 * a_max / r * s))
    best = None
    for pair in _ring_pairs(bounds):
        mean_mm = 0.5 * (pair[0].apothem + pair[1].apothem) / s
        for name, centre, src in cands:
            want = spec.layouts[name].mean_apothem_px / r
            rel = mean_mm / want - 1.0
            if abs(rel) <= LAYOUT_SIZE_TOL and \
                    (best is None or abs(rel) < abs(best[0])):
                best = (rel, pair, name, centre, src)
    if best is None:
        found = ", ".join(f"{2 * b.apothem / s:.2f}mm" for b in bounds) \
            or "nothing hexagonal"
        return HexAnchorReport.refused(
            "full_card", "the ink-cost hexagon could not be found where it "
            f"should be (found {found} across near the top-left corner)")
    rel, pair, name, centre, src = best
    jf = _joint(spec, name, pair)
    x0, y0, off_x, off_y = _mm_from_render(spec, game, centre)
    hx, hy = jf.cx / s, jf.cy / s
    sx, sy = jf.ppm_x / s, jf.ppm_y / s
    rep = HexAnchorReport(
        mode="full_card", layout=name, layout_source=src,
        centre_from_left_mm=hx, centre_from_top_mm=hy,
        expected_from_left_mm=x0, expected_from_top_mm=y0,
        hexagon_px_per_mm={"x": jf.ppm_x, "y": jf.ppm_y},
        size_vs_card_pct={"x": 100.0 * (sx - 1.0), "y": 100.0 * (sy - 1.0)},
        fit_rms_mm=jf.rms_mm, n_points=jf.n)

    bias_unc = game.render_crop_bias_unc_mm or {"x": 0.0, "y": 0.0}
    W, Hc = game.card_w_mm, game.card_h_mm
    aniso = abs(sx / sy - 1.0)

    def one(axis):
        if axis == "x":
            near, far, h, off, dim = "left", "right", hx, off_x, W
            sig = jf.sigma_cx_px / s
        else:
            near, far, h, off, dim = "top", "bottom", hy, off_y, Hc
            sig = jf.sigma_cy_px / s
        stat = math.hypot(sig, _line_stat_mm(lines.get(near), s))
        persp = 0.005 + aniso * h
        w_far = h / dim
        terms = [
            game.edge_def_mm("front", methods.get(near, "step"), axis, s)
            * (1 - w_far),
            game.edge_def_mm("front", methods.get(far, "step"), axis, s)
            * w_far,
            est_extra.get(near, 0.0) * (1 - w_far),
            est_extra.get(far, 0.0) * w_far,
            abs(off) * spec.render_px_per_mm_rel_unc,
            bias_unc.get(axis, 0.0),
            spec.card_size_sd_mm.get(axis, 0.0) * abs(0.5 - w_far),
        ]
        m = _shift_measurement(h - (x0 if axis == "x" else y0),
                               stat, persp, terms)
        if est_extra.get(near) or est_extra.get(far):
            m.status = "estimated"
        return m

    rep.shift_mm = {"x": one("x"), "y": one("y")}
    if not card:
        note = _older_layout_note(spec, name, game)
        if note:
            rep.notes.append(note)
    return rep


# --------------------------------------------------------------------------
# corner close-up
# --------------------------------------------------------------------------

def _ring_template(a, ratio, axis, thick):
    """Edge template for the two ring edges (apothems a and a*ratio): only
    the sides whose normal lies along `axis` (0, 60 or 120 degrees)."""
    R = int(math.ceil(a * ratio * 1.2 + thick + 2))
    t = np.zeros((2 * R + 1, 2 * R + 1), np.float32)
    c = np.array([R, R], float)
    for ap in (a, a * ratio):
        s = 2 * ap / math.sqrt(3)
        for k in (axis, axis + 3):
            n, tg = HF.NORMALS[k], np.array([-HF.NORMALS[k][1],
                                             HF.NORMALS[k][0]])
            p0 = c + n * ap - tg * s / 2
            p1 = c + n * ap + tg * s / 2
            cv2.line(t, tuple(int(round(v)) for v in p0),
                     tuple(int(round(v)) for v in p1), 1.0, thick)
    return t


def locate_hexagon(rgb, max_side=480, min_apothem=MIN_APOTHEM_PX):
    """Rough centre and ring apothems (px) of the most convincing
    upright ink-cost hexagon ring in the photo, or None.

    Oriented-gradient template matching over sizes: for each of the
    hexagon's three side directions, the gradient along that direction is
    matched against a template of just those sides of both ring edges. A
    real hexagon has all three directions, so the weakest one is the
    score. In a close-up the hexagon is big; anything under 2% of the
    photo's shorter side is not considered."""
    h, w = rgb.shape[:2]
    k = min(1.0, max_side / float(max(h, w)))
    small = cv2.resize(rgb, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) \
        if k < 1.0 else rgb
    gx, gy = HF.gradients(cv2.GaussianBlur(small, (3, 3), 0))
    chans = []
    for axis in range(3):
        n = HF.NORMALS[axis]
        g = np.abs(gx * n[0] + gy * n[1])
        chans.append(np.minimum(g, np.percentile(g, 99.5)).astype(np.float32))
    ratio = 1.157
    best = None
    a = max(6.0, 0.9 * min_apothem * k, 0.02 * min(small.shape[:2]))
    while 2 * math.ceil(a * ratio * 1.2 + max(1, round(a / 20)) + 2) + 1 \
            <= min(small.shape[:2]):
        thick = max(1, int(round(a / 20)))
        tot = None
        for axis in range(3):
            t = _ring_template(a, ratio, axis, thick)
            m = cv2.matchTemplate(chans[axis], t, cv2.TM_CCOEFF_NORMED)
            tot = m if tot is None else np.minimum(tot, m)
        j = np.unravel_index(tot.argmax(), tot.shape)
        sc = float(tot[j])
        R = (t.shape[0] - 1) / 2.0
        if best is None or sc > best[0]:
            best = (sc, (j[1] + R) / k, (j[0] + R) / k, a / k)
        a *= 1.04
    if best is None or best[0] < 0.2:
        return None
    _, cx, cy, a0 = best
    return cx, cy, a0, a0 * ratio


def _scan_edge(gray, chroma, side, approx, us, out_px, in_px, ppm, qa):
    """Strict-tier cut edge (brightness step, texture, colour) as in the
    front pipeline. Returns (line or None, report, method)."""
    u_ok, v_ok, diag = E.step_scan(gray, side, approx, us,
                                   search_out_px=out_px, search_in_px=in_px)
    line, rep = _edge_report(side, "step", u_ok, v_ok, diag)
    method = "step"
    if _thin_fit(line, diag):
        u2, v2, d2 = E.texture_scan(gray, side, approx, us,
                                    search_out_px=out_px, search_in_px=in_px)
        line2, rep2 = _edge_report(side, "texture", u2, v2, d2)
        line, rep, diag, method = _prefer_fit(
            (line, rep, diag, "step"), (line2, rep2, d2, "texture"))
    line, rep, diag, method = _colour_edge(
        chroma, side, approx, us, out_px, in_px,
        (line, rep, diag, method), qa, ppm)
    return line, rep, method


def _rotate(rgb, gray, centre, angle_deg):
    M = cv2.getRotationMatrix2D((float(centre[0]), float(centre[1])),
                                angle_deg, 1.0)
    h, w = gray.shape
    rgb2 = cv2.warpAffine(rgb, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)
    gray2 = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC,
                           borderMode=cv2.BORDER_REPLICATE)
    return rgb2, gray2


def _corner_edges(rgb, gray, hx, hy, ppm, x_guess, y_guess, qa, n=40):
    """Left and top cut edges near a hexagon at (hx, hy)."""
    Himg, Wimg = gray.shape
    chroma = E.chromaticity(rgb)
    out_px = max(CORNER_SEARCH_OUT_MM * ppm, E.MIN_SCAN_HALF_WINDOW_PX)
    in_px = max(CORNER_SEARCH_IN_MM * ppm, E.MIN_SCAN_HALF_WINDOW_PX)
    top_v = hy - y_guess * ppm
    left_v = hx - x_guess * ppm
    rows = np.linspace(top_v + CORNER_CLEAR_MM * ppm,
                       min(Himg - 2.0, hy + 30.0 * ppm), n)
    cols = np.linspace(left_v + CORNER_CLEAR_MM * ppm,
                       min(Wimg - 2.0, hx + 45.0 * ppm), n)
    out = {}
    for side, approx, us in (("left", left_v, rows), ("top", top_v, cols)):
        # the search may run off the photo: a close-up often has only a
        # thin strip of background, so clip it, down to a minimum
        o_px = min(out_px, approx - 1.0)
        if o_px < max(0.5 * ppm, 4.0) or len(us) < 2 \
                or us[-1] - us[0] < 3 * ppm:
            out[side] = (None, None, None,
                         f"the {side} edge of the card is too close to the "
                         "edge of the photo, or not in it")
            continue
        line, rep, method = _scan_edge(gray, chroma, side, approx, us,
                                       o_px, in_px, ppm, qa)
        why = None if line is not None else (
            f"the {side} edge of the card could not be measured: "
            + ("; ".join(rep.notes) or "no clear edge"))
        out[side] = (line, rep, method, why)
    return out


def _edge_qa(img_rgb, img_gray, h):
    """Warnings the chosen hypothesis's edge scans raised (colour edge
    adopted and the like), re-run once so they are not duplicated per
    hypothesis."""
    qa = []
    _corner_edges(img_rgb, img_gray, h["jf"].cx, h["jf"].cy, h["ppm"],
                  h["x0"], h["y0"], qa)
    return qa


def _corner_refuse(res, reason, advice=RESHOOT_CORNER):
    res.hex_check = HexAnchorReport.refused("corner", reason, advice)
    res.shift_mm = {"x": Measurement.refused("mm", reason, advice),
                    "y": Measurement.refused("mm", reason, advice)}
    res.equivalent_ratio_lr = Ratio.refused("LR", reason, advice)
    res.equivalent_ratio_tb = Ratio.refused("TB", reason, advice)
    return res


def equivalent_ratio(axis, m: Measurement, total_margin) -> Ratio:
    """Grading-style ratio from a print shift (same convention as the
    official-picture path)."""
    if m.status == "refused":
        return Ratio.refused(axis, m.refusal_reason, m.refusal_advice)
    a = total_margin / 2.0 + m.value
    b = total_margin / 2.0 - m.value
    if a <= 0 or b <= 0:
        return Ratio.refused(
            axis, "The print is shifted further than a normal border is "
                  "wide, so a centring ratio would not mean anything here. "
                  "Use the millimetre figure instead.")
    k = 100.0 / total_margin
    s = m.uncertainty
    return Ratio(axis=axis, first_pct=100.0 * a / (a + b),
                 uncertainty_pts=Uncertainty(
                     statistical=s.statistical * k,
                     perspective=s.perspective * k,
                     edge_definition=s.edge_definition * k),
                 status=m.status)


def analyze_corner(photo, game: GameSpec, card_id: Optional[str] = None,
                   render_source=None, out_dir: Optional[str] = None,
                   make_overlay: bool = True) -> BorderlessResult:
    """Front print shift from a close-up of the card's top-left corner.

    Needs the whole ink-cost hexagon and a strip of background beyond the
    top and left edges. The hexagon's printed size is the ruler."""
    rgb, gray, inp = load_photo(photo)
    res = BorderlessResult(kind="corner", game=game.name, input=inp,
                           tilt=TiltReport(corrected=False),
                           method="ink_hexagon")
    qa = res.qa
    spec = game.hex_anchor
    if spec is None:
        return _corner_refuse(res, "There is no ink-cost hexagon layout on "
                              "record for this game.", None)

    card, db_dir = None, None
    if card_id:
        if render_source is None:
            from .games.lorcana import LorcanaRenderSource
            render_source = LorcanaRenderSource()
        card = render_source.resolve(card_id)
        db_dir = getattr(render_source, "local_db_dir", None)
    else:
        from .games.lorcana import _default_local_db_dir
        db_dir = _default_local_db_dir()
    try:
        cands = candidate_layouts(spec, card, db_dir)
    except ValueError as err:
        return _corner_refuse(res, f"This card cannot be checked this way: "
                              f"{err}.", None)
    if card is None:
        cands = [c for c in cands if c[0] in spec.default_layouts]

    coarse = locate_hexagon(rgb)
    if coarse is None:
        return _corner_refuse(
            res, "No ink-cost hexagon could be found in this photo.")
    cx0, cy0, a_lo, a_hi = coarse
    if a_lo < MIN_APOTHEM_PX:
        return _corner_refuse(
            res, f"The hexagon is only {2 * a_lo:.0f} pixels across - too "
                 "small to measure. Move closer or zoom in.")

    def fit_bounds(img, cx, cy):
        return HF.find_boundaries(img, (cx, cy), max(3.0, 0.12 * a_lo),
                                  (0.8 * a_lo, 1.25 * a_hi))

    img_rgb, img_gray = rgb, gray
    angle_total = 0.0
    result = None
    for attempt in range(2):
        bounds = fit_bounds(img_rgb, cx0, cy0)
        pairs = _ring_pairs(bounds)
        if not pairs:
            return _corner_refuse(
                res, "The ink-cost hexagon was found, but its ring could "
                     "not be measured cleanly.")
        inner, outer = max(pairs, key=lambda p: p[0].n + p[1].n)
        hyps, why_bad = [], []
        for name, centre, src in cands:
            jf = _joint(spec, name, (inner, outer))
            ppm = 0.5 * (jf.ppm_x + jf.ppm_y)
            x0, y0, off_x, off_y = _mm_from_render(spec, game, centre)
            h = dict(name=name, centre=centre, src=src, jf=jf, ppm=ppm,
                     x0=x0, y0=y0, off=(off_x, off_y))
            # each layout implies its own scale, so its own edge positions
            h["edges"] = _corner_edges(img_rgb, img_gray, jf.cx, jf.cy, ppm,
                                       x0, y0, [])
            bad = [why for (_, _, _, why) in h["edges"].values() if why]
            if bad:
                why_bad.append((name, bad))
                continue
            hyps.append(h)
        if not hyps:
            return _corner_refuse(res, " ".join(
                b[0].upper() + b[1:] + "." for b in why_bad[0][1]))
        edges = hyps[0]["edges"]
        L, T = edges["left"][0], edges["top"][0]
        a_l = -math.degrees(math.atan(L.m))
        a_t = math.degrees(math.atan(T.m))
        rot = 0.5 * (a_l + a_t)
        if attempt == 0 and abs(rot) > 0.25:
            angle_total += rot
            img_rgb, img_gray = _rotate(rgb, gray, (cx0, cy0), angle_total)
            continue
        result = (hyps, edges, a_l, a_t)
        break
    if result is None:
        return _corner_refuse(res, "The photo is turned too far to "
                              "straighten reliably. Line the card up with "
                              "the photo's edges.")
    hyps, edges, a_l, a_t = result
    square_dev = a_t - a_l
    res.corner_angles_deg = {"top_left": 90.0 + square_dev}
    res.tilt.notes.append(
        f"the photo was turned {angle_total:+.2f} degrees to straighten "
        "the card" if angle_total else "the card was already straight")
    if abs(square_dev) > CORNER_SQUARE_DEG:
        return _corner_refuse(
            res, f"The corner comes out {90 + square_dev:.1f} degrees "
                 "instead of 90, so the photo was taken at an angle. A "
                 "close-up has no other edges to correct for that.")
    for h in hyps:
        jf = h["jf"]
        Lh, Th = h["edges"]["left"][0], h["edges"]["top"][0]
        dl = (jf.cx - float(Lh.v_at(jf.cy))) / jf.ppm_x
        dt = (jf.cy - float(Th.v_at(jf.cx))) / jf.ppm_y
        h.update(dl=dl, dt=dt, sx=dl - h["x0"], sy=dt - h["y0"])
    plausible = [h for h in hyps
                 if max(abs(h["sx"]), abs(h["sy"])) <= PLAUSIBLE_SHIFT_MM]
    all_b = fit_bounds(img_rgb, hyps[0]["jf"].cx, hyps[0]["jf"].cy)
    inner_b = min(all_b, key=lambda b: b.apothem) if all_b else None
    swirl = _swirl_seen(all_b, inner_b) if all_b else False
    looks = "inkable" if swirl else "uninkable"
    unsearched = {n for n, _ in why_bad}
    if len(plausible) == 1 and looks in unsearched:
        return _corner_refuse(
            res, f"The hexagon looks like a{'n' if looks[0] in 'aeiou' else ''}"
                 f" {looks} card's ({'with' if swirl else 'without'} the "
                 "flower-shaped surround), but only the "
                 f"{plausible[0]['name']} design could be checked against "
                 "the card edges - the photo is cropped too tight to look "
                 f"for them where the {looks} design puts them.")
    if not plausible:
        return _corner_refuse(
            res, "The hexagon and the card edges do not fit together for "
                 "any known layout: " + "; ".join(
                     f"as {h['name']} the print would be "
                     f"{h['sx']:+.2f}mm across and {h['sy']:+.2f}mm down"
                     for h in hyps) + ". An edge was probably found in the "
                 "wrong place.")
    if len(plausible) > 1:
        def evidence(o):
            e = -(o["sx"] ** 2 + o["sy"] ** 2) / (2 * SHIFT_PRIOR_MM ** 2)
            return e + (SWIRL_VOTE if o["name"] == looks else -SWIRL_VOTE)
        h = max(plausible, key=evidence)
        qa.append(QAFlag(
            "HEX_LAYOUT_CHOSEN",
            "Two hexagon designs fit this photo. The "
            f"{h['name']} one was used: it puts the print nearer the "
            "centre, which is where it usually is"
            + (", and the hexagon looks like it" if h["name"] == looks
               else ", although the hexagon looks more like the other")
            + ". Read as the other design the shift would be " + ", ".join(
                f"{o['sx']:+.2f}mm across and {o['sy']:+.2f}mm down"
                for o in plausible if o is not h) + "."))
    else:
        h = plausible[0]

    qa.extend(_edge_qa(img_rgb, img_gray, h))
    jf = h["jf"]
    L, repL, methL, _ = h["edges"]["left"]
    T, repT, methT, _ = h["edges"]["top"]
    res.edge_fits = [repL, repT]
    rep = HexAnchorReport(
        mode="corner", layout=h["name"], layout_source=h["src"],
        centre_from_left_mm=h["dl"], centre_from_top_mm=h["dt"],
        expected_from_left_mm=h["x0"], expected_from_top_mm=h["y0"],
        hexagon_px_per_mm={"x": jf.ppm_x, "y": jf.ppm_y},
        fit_rms_mm=jf.rms_mm, n_points=jf.n)
    inp.px_per_mm = h["ppm"]
    aniso = abs(jf.ppm_x / jf.ppm_y - 1.0)
    bias_unc = game.render_crop_bias_unc_mm or {"x": 0.0, "y": 0.0}

    def one(axis):
        if axis == "x":
            d, line, meth, sig_px, ppm, rel = (h["dl"], L, methL,
                                               jf.sigma_cx_px, jf.ppm_x,
                                               jf.sigma_ppm_x_rel)
            off = h["off"][0]
        else:
            d, line, meth, sig_px, ppm, rel = (h["dt"], T, methT,
                                               jf.sigma_cy_px, jf.ppm_y,
                                               jf.sigma_ppm_y_rel)
            off = h["off"][1]
        stat = math.sqrt((sig_px / ppm) ** 2 + _line_stat_mm(line, ppm) ** 2
                         + (d * rel) ** 2)
        persp = (0.005 + aniso * d / 2.0
                 + 0.25 * d * abs(math.sin(math.radians(square_dev))))
        terms = [
            game.edge_def_mm("front", meth, axis, ppm),
            (d + abs(off)) * spec.render_px_per_mm_rel_unc,
            d * spec.size_rel_unc,
            bias_unc.get(axis, 0.0),
            spec.card_size_sd_mm.get(axis, 0.0) / 2.0,
        ]
        return _shift_measurement(d - (h["x0"] if axis == "x" else h["y0"]),
                                  stat, persp, terms)

    rep.shift_mm = {"x": one("x"), "y": one("y")}
    if card is None:
        note = _older_layout_note(spec, h["name"], game)
        if note:
            rep.notes.append(note)
            qa.append(QAFlag("HEX_LAYOUT_ASSUMED", note, severity="info"))
    res.hex_check = rep
    res.shift_mm = dict(rep.shift_mm)
    res.equivalent_ratio_lr = equivalent_ratio(
        "LR", res.shift_mm["x"], game.equiv_margin_lr_mm)
    res.equivalent_ratio_tb = equivalent_ratio(
        "TB", res.shift_mm["y"], game.equiv_margin_tb_mm)

    if make_overlay:
        res.overlay = _corner_overlay(img_rgb, L, T, jf, h, spec, res,
                                      photo, out_dir)
    return res


def _draw_hexagon(img, cx, cy, a, color, lw):
    pts = []
    for k in range(6):
        ang = math.radians(-90 + 60 * k)
        rr = a / math.cos(math.radians(30))
        pts.append((cx + rr * math.cos(ang), cy + rr * math.sin(ang)))
    cv2.polylines(img, [np.array(pts, np.int32)], True, color, lw)


def _corner_overlay(img_rgb, L, T, jf, h, spec, res, photo, out_dir):
    from .overlay import C_EDGE, C_FRAME, Overlay
    ov = Overlay(img_rgb)
    ov.line(L, C_EDGE)
    ov.line(T, C_EDGE)
    for ring in spec.layouts[h["name"]].rings:
        a = 0.5 * (ring[2] + ring[3]) / spec.render_px_per_mm * \
            0.5 * (jf.ppm_x + jf.ppm_y)
        _draw_hexagon(ov.img, jf.cx, jf.cy, a, C_FRAME, max(1, ov.lw // 2))
    # where the hexagon would be on a centred card
    ex = float(L.v_at(jf.cy)) + h["x0"] * jf.ppm_x
    ey = float(T.v_at(jf.cx)) + h["y0"] * jf.ppm_y
    cv2.drawMarker(ov.img, (int(round(ex)), int(round(ey))), C_EDGE,
                   cv2.MARKER_CROSS, max(8, int(0.6 * jf.ppm_x)), ov.lw)
    sx, sy = res.shift_mm["x"], res.shift_mm["y"]
    ov.banner([
        f"CORNER print shift: x {sx.value:+.2f}mm  y {sy.value:+.2f}mm",
        "(+x = print sits toward the right edge, +y = toward bottom)",
        f"same as L/R {res.equivalent_ratio_lr.display or 'not measured'}"
        f"  T/B {res.equivalent_ratio_tb.display or 'not measured'}",
        f"ruler: the {h['name']} ink hexagon; cross = where it sits on a "
        "centred card"])
    out = Path(out_dir) if out_dir else Path(photo).parent
    out.mkdir(parents=True, exist_ok=True)
    return ov.save(out / (Path(photo).stem + "_corner_overlay.jpg"))
