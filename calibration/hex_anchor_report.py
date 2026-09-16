"""Summarise card_db/hex_anchors_percard.csv into card_db/hex_anchors_report.json.

Groups cards into hexagon families by their boundary apothems, takes the centre
from the two inner boundaries only (the outer swirl outline on inkable cards
sits ~0.4px off in x because the swirl interrupts its vertical sides), and
reports the per-family centre, spread and size in render px and in mm.

    python calibration/hex_anchor_report.py
"""
import csv, json, os, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "..", "card_db")
RENDER_W, RENDER_H = 1468, 2048
CARD_W, CARD_H = 62.9, 87.9                    # manufactured size (CLAUDE.md)
PPM_X, PPM_Y = RENDER_W / CARD_W, RENDER_H / CARD_H

FAMILIES = {                                   # name: (inner apothem, second apothem)
    "inkable": (57.74, 66.62),
    "uninkable": (67.72, 78.56),
    "uninkable_large": (75.69, 87.68),
}


def family(ap):
    for name, ks in FAMILIES.items():
        idx = []
        for k in ks:
            j = [i for i, a in enumerate(ap) if abs(a - k) < 1.2]
            if not j:
                break
            idx.append(j[0])
        else:
            return name, idx
    return None, None


def stats(v):
    v = np.asarray(v, float); med = float(np.median(v))
    return {"n": len(v), "median": round(med, 3),
            "robust_sd": round(float(1.4826 * np.median(np.abs(v - med))), 3),
            "sd": round(float(v.std()), 3),
            "pct_within_0p5px": round(float(np.mean(np.abs(v - med) < 0.5) * 100), 2)}


def main():
    H = list(csv.DictReader(open(os.path.join(DB, "hex_anchors_percard.csv"), newline="")))
    A = {r["id"]: r for r in csv.DictReader(open(os.path.join(DB, "anchors_percard.csv"), newline=""))}
    groups = collections.defaultdict(list); unmatched = []
    for r in H:
        if int(r["n_bounds"] or 0) <= 0:
            unmatched.append(r["id"]); continue
        ap = [float(x) for x in r["apothems"].split(";")]
        fam, idx = family(ap)
        if fam is None:
            unmatched.append(r["id"]); continue
        cxs = [float(x) for x in r["cx_each"].split(";")]; cys = [float(x) for x in r["cy_each"].split(";")]
        cx, cy = np.mean([cxs[i] for i in idx]), np.mean([cys[i] for i in idx])
        setc = r["id"].split("-")[0]
        # sub-groups found by the survey (layout changes, not crop: frame edges and
        # footer emblem do not move with them)
        if fam == "uninkable" and setc in ("1", "2", "3"):
            fam = "uninkable_sets1to3"
        if setc == "Q2":
            fam += "_Q2promo"
        groups[fam].append((r["id"], cx, cy, [ap[i] for i in idx], r.get("search", "open")))
    rep = {"render_px_per_mm": {"x": round(PPM_X, 4), "y": round(PPM_Y, 4)},
           "card_mm": [CARD_W, CARD_H], "families": {}}
    for fam, rows in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        cx = np.array([x[1] for x in rows]); cy = np.array([x[2] for x in rows])
        mx, my = np.median(cx), np.median(cy)
        off = [x[0] for x in rows if abs(x[1] - mx) > 1 or abs(x[2] - my) > 1]
        a0 = [x[3][0] for x in rows]; a1 = [x[3][1] for x in rows]
        rep["families"][fam] = {
            "n": len(rows), "n_window_search": sum(x[4] == "window" for x in rows),
            "cx_px": stats(cx), "cy_px": stats(cy),
            "apothem_inner_px": stats(a0), "apothem_second_px": stats(a1),
            "centre_from_left_mm": round(float(mx / PPM_X), 4),
            "centre_from_top_mm": round(float(my / PPM_Y), 4),
            "centre_from_card_centre_mm": [round(float((mx - RENDER_W / 2) / PPM_X), 4),
                                           round(float((my - RENDER_H / 2) / PPM_Y), 4)],
            "inner_flat_to_flat_mm": round(float(2 * np.median(a0) / PPM_X), 4),
            "cards_off_by_gt_1px": off,
            "by_bordered_rarity": {
                f"{b}|{ra}": {"n": len(v), "cx": round(float(np.median([x[1] for x in v])), 3),
                              "cy": round(float(np.median([x[2] for x in v])), 3)}
                for (b, ra), v in sorted(_group(rows, A).items())},
        }
    rep["unmatched"] = unmatched
    rep["notes"] = ("Render geometry only: this shows the hexagon is a fixed DIGITAL layout "
                    "constant per family. Mapping render px to a physical cut carries the "
                    "render_crop_bias_mm uncertainty (see calibration/NOTES.md). Physical print "
                    "registration of the hexagon still needs checking on real scans.")
    out = os.path.join(DB, "hex_anchors_report.json")
    json.dump(rep, open(out, "w"), indent=2)
    for fam, f in rep["families"].items():
        print(f"{fam:28s} n={f['n']:4d} (window {f['n_window_search']:2d}) "
              f"cx {f['cx_px']['median']:.3f} rsd {f['cx_px']['robust_sd']:.3f} sd {f['cx_px']['sd']:.3f} | "
              f"cy {f['cy_px']['median']:.3f} rsd {f['cy_px']['robust_sd']:.3f} sd {f['cy_px']['sd']:.3f} | "
              f"L {f['centre_from_left_mm']:.3f}mm T {f['centre_from_top_mm']:.3f}mm "
              f"f2f {f['inner_flat_to_flat_mm']:.3f}mm | off>1px {len(f['cards_off_by_gt_1px'])} {f['cards_off_by_gt_1px'][:8]}")
    print("unmatched", len(unmatched), unmatched)


def _group(rows, A):
    g = collections.defaultdict(list)
    for x in rows:
        a = A.get(x[0], {})
        g[(a.get("bordered_empirical", "?"), a.get("rarity", "?"))].append(x)
    return g


if __name__ == "__main__":
    main()
