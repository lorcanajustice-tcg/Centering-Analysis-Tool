"""The ink-cost hexagon check, on drawn (synthetic) cards only.

No card art is used: the card is a dark rectangle and the hexagon is drawn
from the layout constants in games/lorcana.py, so the tests check the
geometry and the arithmetic, not the survey itself.
"""
import csv

import cv2
import numpy as np
import pytest

from centering import geometry as G
from centering import hexanchor as HA
from centering import hexfit as HF
from centering.games.lorcana import LORCANA
from centering.plain import FLAGS

SPEC = LORCANA.hex_anchor
W, H = LORCANA.card_w_mm, LORCANA.card_h_mm
SS = 4  # supersampling for anti-aliased drawing


def _hexagon(cx, cy, a_v, a_s, sx, sy):
    """Vertices (px) of a hexagon whose vertical sides sit a_v mm and whose
    slanted sides sit a_s mm from the centre, at sx/sy px per mm."""
    A = [a_v if k % 3 == 0 else a_s for k in range(6)]
    pts = []
    for k in range(6):
        n1, n2 = HF.NORMALS[k], HF.NORMALS[(k + 1) % 6]
        v = np.linalg.solve(np.array([n1, n2]), np.array([A[k], A[(k + 1) % 6]]))
        pts.append((cx + v[0] * sx, cy + v[1] * sy))
    return np.array(pts)


def _poly(img, pts, colour):
    # output pixel j covers supersampled pixels SS*j .. SS*j+SS-1, whose
    # centre is at SS*j + (SS-1)/2
    q = pts * SS + (SS - 1) / 2.0
    cv2.fillPoly(img, [np.round(q * 16).astype(np.int32)], colour,
                 lineType=cv2.LINE_8, shift=4)


def draw_card(layout="inkable", shift=(0.0, 0.0), ppm=20.0, pad_mm=6.0,
              stretch_y=1.0, region_mm=None):
    """Top-left part of a card (region_mm across and down, default the
    whole card) on a light background. Returns (rgb, card corner px,
    ppm_x, ppm_y)."""
    lay = SPEC.layouts[layout]
    sx, sy = ppm, ppm * stretch_y
    reg_w, reg_h = region_mm or (W, H)
    w = int((pad_mm + reg_w) * sx)
    h = int((pad_mm + reg_h) * sy)
    img = np.full((h * SS, w * SS, 3), 235, np.uint8)
    x0, y0 = pad_mm * sx + 0.37, pad_mm * sy + 0.61     # off the pixel grid
    card = np.array([[x0, y0], [x0 + W * sx, y0], [x0 + W * sx, y0 + H * sy],
                     [x0, y0 + H * sy]])
    _poly(img, card, (40, 60, 90))
    ex, ey, _, _ = HA._mm_from_render(SPEC, LORCANA, lay.centre_px)
    cx = x0 + (ex + shift[0]) * sx
    cy = y0 + (ey + shift[1]) * sy
    r = SPEC.render_px_per_mm
    inner, outer = lay.rings
    a_in = 0.5 * (inner[2] + inner[3]) / r
    if lay.swirl:   # gold surround with its dark hexagonal outline
        _poly(img, _hexagon(cx, cy, 1.9 * a_in, 1.9 * a_in, sx, sy),
              (200, 170, 100))
        _poly(img, _hexagon(cx, cy, 1.248 * a_in, 1.248 * a_in, sx, sy),
              (15, 15, 15))
    else:           # plain hexagon with a dark outline
        _poly(img, _hexagon(cx, cy, 1.08 * outer[2] / r, 1.08 * outer[3] / r,
                            sx, sy), (15, 15, 15))
    _poly(img, _hexagon(cx + outer[0] / r * sx, cy + outer[1] / r * sy,
                        outer[2] / r, outer[3] / r, sx, sy), (205, 175, 105))
    _poly(img, _hexagon(cx + inner[0] / r * sx, cy + inner[1] / r * sy,
                        inner[2] / r, inner[3] / r, sx, sy), (20, 20, 25))
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return img, card, sx, sy


def _lines(card):
    tl, tr, br, bl = card
    ys = np.linspace(tl[1] + 50, bl[1] - 50, 20)
    xs = np.linspace(tl[0] + 50, tr[0] - 50, 20)
    return {
        "left": G.FittedLine.fit("v", ys, np.full_like(ys, tl[0])),
        "right": G.FittedLine.fit("v", ys, np.full_like(ys, tr[0])),
        "top": G.FittedLine.fit("h", xs, np.full_like(xs, tl[1])),
        "bottom": G.FittedLine.fit("h", xs, np.full_like(xs, bl[1])),
    }


# --- hexfit ---------------------------------------------------------------

def test_hexfit_recovers_centre_and_axis_scales():
    img, card, sx, sy = draw_card("uninkable", ppm=24.0, stretch_y=1.02,
                                  region_mm=(16, 16))
    lay = SPEC.layouts["uninkable"]
    ex, ey, _, _ = HA._mm_from_render(SPEC, LORCANA, lay.centre_px)
    cx, cy = card[0][0] + ex * sx, card[0][1] + ey * sy
    b = HF.find_boundaries(img, (cx + 5, cy - 4), 12, (50, 90))
    pairs = HA._ring_pairs(b)
    assert pairs
    jf = HA._joint(SPEC, "uninkable", pairs[0])
    assert abs(jf.cx - cx) < 0.1 and abs(jf.cy - cy) < 0.1
    assert jf.ppm_x == pytest.approx(sx, rel=0.004)
    assert jf.ppm_y == pytest.approx(sy, rel=0.004)


# --- layouts --------------------------------------------------------------

def test_centred_positions_follow_the_survey():
    # inkable centre 146.560, 156.788 px; 23.64 px/mm; crop bias y -0.08
    ex, ey, ox, oy = HA._mm_from_render(SPEC, LORCANA,
                                        SPEC.layouts["inkable"].centre_px)
    assert ex == pytest.approx(W / 2 + (146.560 - 734) / 23.64, abs=1e-6)
    assert ey == pytest.approx(H / 2 - 0.08 + (156.788 - 1024) / 23.64,
                               abs=1e-6)
    assert 6.55 < ex < 6.65 and 7.15 < ey < 7.22


def test_unknown_card_assumes_recent_sets():
    c = HA.candidate_layouts(SPEC, None)
    assert [x[0] for x in c] == ["inkable", "uninkable"]
    assert c[1][1] == SPEC.layouts["uninkable"].centre_px
    note = HA._older_layout_note(SPEC, "uninkable", LORCANA)
    assert "sets 1, 2, 3" in note and "0.14mm" in note
    assert HA._older_layout_note(SPEC, "inkable", LORCANA) is None


def test_known_old_set_uses_the_older_layout(tmp_path):
    c = HA.candidate_layouts(SPEC, {"setCode": "2", "number": 5}, tmp_path)
    old = dict((n, ctr) for n, ctr, _ in c)["uninkable"]
    assert old == (149.598, 155.567)


def _write_rows(path, rows):
    fields = ["id", "w", "h", "search", "n_bounds", "cx", "cy", "apothems",
              "fit_rms", "strengths", "cx_each", "cy_each"]
    with open(path, "w", newline="") as f:
        wr = csv.DictWriter(f, fields, restval="")
        wr.writeheader()
        for r in rows:
            wr.writerow(r)


def test_known_card_uses_its_own_survey_row(tmp_path):
    _write_rows(tmp_path / SPEC.percard_csv, [
        {"id": "9-9", "n_bounds": 3, "apothems": "57.7;66.6;72.0",
         "cx_each": "147.0;147.2;147.9", "cy_each": "156.5;156.7;156.9"},
        {"id": "11-241", "n_bounds": 2, "apothems": "55.2;63.9",
         "cx_each": "146.5;146.5", "cy_each": "156.1;156.1"},
    ])
    HA._percard_rows.cache_clear()
    c = HA.candidate_layouts(SPEC, {"setCode": "9", "number": 9}, tmp_path)
    assert c == [("inkable", (147.1, 156.6), c[0][2])]
    with pytest.raises(ValueError, match="special layout"):
        HA.candidate_layouts(SPEC, {"setCode": "11", "number": 241},
                             tmp_path)
    HA._percard_rows.cache_clear()


# --- full card ------------------------------------------------------------

@pytest.mark.parametrize("layout,shift", [("inkable", (0.30, -0.20)),
                                          ("uninkable", (-0.45, 0.25))])
def test_full_card_recovers_the_print_shift(layout, shift, tmp_path):
    img, card, sx, _ = draw_card(layout, shift, ppm=20.0)
    methods = {s: "step" for s in ("left", "right", "top", "bottom")}
    rep = HA.measure_in_card(img, card, LORCANA, sx, _lines(card), methods,
                             {}, card=None, db_dir=tmp_path)
    assert rep.status == "measured", rep.refusal_reason
    assert rep.layout == layout
    assert rep.shift_mm["x"].value == pytest.approx(shift[0], abs=0.02)
    assert rep.shift_mm["y"].value == pytest.approx(shift[1], abs=0.02)
    for ax in "xy":
        assert abs(rep.size_vs_card_pct[ax]) < 0.5
        u = rep.shift_mm[ax].uncertainty
        # the calibration terms alone put a floor under the margin
        assert u.total > 0.05
    if layout == "uninkable":
        assert any("sets 1, 2, 3" in n for n in rep.notes)


def test_full_card_refuses_when_there_is_no_hexagon(tmp_path):
    img, card, sx, _ = draw_card()
    img[:int(20 * sx), :int(20 * sx)] = (40, 60, 90)
    img[:int(card[0][1]), :] = 235
    img[:, :int(card[0][0])] = 235
    methods = {s: "step" for s in ("left", "right", "top", "bottom")}
    rep = HA.measure_in_card(img, card, LORCANA, sx, _lines(card), methods,
                             {}, db_dir=tmp_path)
    assert rep.status == "refused"
    assert "could not be found" in rep.refusal_reason


# --- corner close-up ------------------------------------------------------

@pytest.mark.parametrize("layout,shift", [("inkable", (-0.25, 0.35)),
                                          ("uninkable", (0.20, -0.30))])
def test_corner_recovers_the_print_shift(layout, shift, tmp_path):
    img, card, _, _ = draw_card(layout, shift, ppm=30.0, pad_mm=4.0,
                                region_mm=(26, 26))
    p = tmp_path / "corner.png"
    cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    r = HA.analyze_corner(p, LORCANA, make_overlay=True, out_dir=tmp_path)
    assert r.kind == "corner" and r.method == "ink_hexagon"
    assert r.hex_check.layout == layout, r.hex_check.refusal_reason
    assert r.shift_mm["x"].value == pytest.approx(shift[0], abs=0.05)
    assert r.shift_mm["y"].value == pytest.approx(shift[1], abs=0.05)
    assert r.hex_check.hexagon_px_per_mm["x"] == pytest.approx(30.0, rel=0.01)
    assert r.equivalent_ratio_lr.status == "measured"
    assert r.overlay and (tmp_path / "corner_corner_overlay.jpg").exists()
    d = r.to_dict()
    assert d["kind"] == "corner" and d["result"]["method"] == "ink_hexagon"


def test_corner_straightens_a_turned_photo(tmp_path):
    img, card, _, _ = draw_card("inkable", (0.1, -0.1), ppm=30.0, pad_mm=6.0,
                                region_mm=(26, 26))
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 1.5, 1.0)
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                         borderValue=(235, 235, 235))
    p = tmp_path / "turned.png"
    cv2.imwrite(str(p), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    r = HA.analyze_corner(p, LORCANA, make_overlay=False)
    assert r.shift_mm["x"].value == pytest.approx(0.1, abs=0.06)
    assert r.shift_mm["y"].value == pytest.approx(-0.1, abs=0.06)
    assert "turned" in r.tilt.notes[0]


def test_corner_refuses_without_a_hexagon(tmp_path):
    img = np.full((600, 600, 3), 235, np.uint8)
    img[100:, 100:] = (40, 60, 90)
    p = tmp_path / "blank.png"
    cv2.imwrite(str(p), img)
    r = HA.analyze_corner(p, LORCANA, make_overlay=False)
    assert r.shift_mm["x"].status == "refused"
    assert r.equivalent_ratio_tb.status == "refused"


def test_corner_refuses_when_the_edges_are_cropped_off(tmp_path):
    img, card, _, _ = draw_card("inkable", ppm=30.0, pad_mm=4.0,
                                region_mm=(26, 26))
    x0 = int(card[0][0] + 1.5 * 30)
    p = tmp_path / "tight.png"
    cv2.imwrite(str(p), cv2.cvtColor(img[:, x0:], cv2.COLOR_RGB2BGR))
    r = HA.analyze_corner(p, LORCANA, make_overlay=False)
    assert r.shift_mm["x"].status == "refused"
    assert "left edge" in r.shift_mm["x"].refusal_reason


# --- small pieces ---------------------------------------------------------

def test_equivalent_ratio_convention():
    from centering.types import Measurement, Uncertainty
    m = Measurement(0.2, "mm", Uncertainty(0.01, 0.0, 0.05))
    r = HA.equivalent_ratio("LR", m, 4.0)
    assert r.first_pct == pytest.approx(55.0)
    assert r.uncertainty_pts.edge_definition == pytest.approx(1.25)
    big = Measurement(3.0, "mm", Uncertainty())
    assert HA.equivalent_ratio("LR", big, 4.0).status == "refused"


def test_every_hexagon_warning_has_plain_wording():
    for code in ("HEX_ANCHOR_USED", "HEX_ANCHOR_DISAGREES",
                 "HEX_SCALE_MISMATCH", "HEX_LAYOUT_ASSUMED",
                 "HEX_LAYOUT_CHOSEN", "HEX_CHECK_SKIPPED"):
        assert code in FLAGS
        assert "_" not in FLAGS[code][0]


def test_front_without_a_card_id_is_measured_from_the_hexagon(tmp_path):
    from centering.borderless import analyze_borderless
    img, card, _, _ = draw_card("inkable", (-0.35, 0.15), ppm=16.0,
                                pad_mm=8.0)
    full = np.full((img.shape[0] + int(8 * 16), img.shape[1] + int(8 * 16),
                    3), 235, np.uint8)
    full[:img.shape[0], :img.shape[1]] = img
    p = tmp_path / "front.png"
    cv2.imwrite(str(p), cv2.cvtColor(full, cv2.COLOR_RGB2BGR))
    r = analyze_borderless(p, None, LORCANA, out_dir=str(tmp_path))
    assert r.method == "ink_hexagon" and r.render is None
    assert r.shift_mm["x"].value == pytest.approx(-0.35, abs=0.05)
    assert r.shift_mm["y"].value == pytest.approx(0.15, abs=0.05)
    assert r.equivalent_ratio_lr.first_pct < 50
    assert any(q.code == "HEX_ANCHOR_USED" for q in r.qa)
    d = r.to_dict()
    assert d["schema_version"] == "1.2"
    assert d["result"]["hex_check"]["layout"] == "inkable"
    assert r.overlay
