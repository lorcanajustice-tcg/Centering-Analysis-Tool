"""Ink-cost hexagon survey over the local render database.

For every render in card_db/images, locate the pointy-top ink-cost hexagon in
the top-left corner and record its centre and the apothem (centre-to-side
distance) of every hexagonal boundary found, all subpixel, in render pixels.
Output: card_db/hex_anchors_percard.csv (numbers only). Resumable: rows already
present are skipped, so it can be run in short chunks:

    python calibration/hex_anchor_survey.py --limit 150 --workers 4

The question it answers: is the hexagon's position a fixed print-layout
constant (per frame family), tight enough to act as a card-agnostic centering
anchor? Summarise with calibration/hex_anchor_report.py.
"""
import argparse, csv, glob, os, sys, time
from multiprocessing import Pool
import cv2, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "..", "card_db")
OUT = os.path.join(DB, "hex_anchors_percard.csv")

ANG = np.deg2rad(np.arange(6) * 60.0)
NORM = np.stack([np.cos(ANG), np.sin(ANG)], 1)   # side k satisfies n_k.(p - c) = a
TANG = np.stack([-NORM[:, 1], NORM[:, 0]], 1)
T = np.linspace(-0.45, 0.45, 13)


def grad_img(bgr):
    """Per-pixel gradient from whichever Lab channel is strongest (rings are gold, white, orange...)."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    gx = np.zeros(lab.shape[:2], np.float32); gy = gx.copy(); best = gx.copy()
    for c in range(3):
        x = cv2.Scharr(lab[..., c], cv2.CV_32F, 1, 0); y = cv2.Scharr(lab[..., c], cv2.CV_32F, 0, 1)
        m = x * x + y * y; sel = m > best
        best[sel] = m[sel]; gx[sel] = x[sel]; gy[sel] = y[sel]
    return gx, gy


def _score(gx, gy, cx, cy, a):
    s = 2 * a / np.sqrt(3); tot = 0
    for k in range(6):
        px = cx + NORM[k, 0] * a + TANG[k, 0] * s * T[:, None, None, None]
        py = cy + NORM[k, 1] * a + TANG[k, 1] * s * T[:, None, None, None]
        xi = np.clip(np.rint(px).astype(int), 0, gx.shape[1] - 1)
        yi = np.clip(np.rint(py).astype(int), 0, gx.shape[0] - 1)
        g = np.abs(gx[yi, xi] * NORM[k, 0] + gy[yi, xi] * NORM[k, 1])
        tot = tot + np.minimum(g, 3000).mean(0)
    return tot / 6


def refine(GX, GY, cx, cy, a, half=3.0, nt=31):
    s = 2 * a / np.sqrt(3); rows, rhs, sides = [], [], []
    offs = np.arange(-half, half + 0.01, 0.25)
    for k in range(6):
        n, t = NORM[k], TANG[k]
        for tt in np.linspace(-0.38, 0.38, nt):
            pts = (np.array([cx, cy]) + n * a + t * s * tt)[None, :] + offs[:, None] * n[None, :]
            mx, my = pts[:, 0].astype(np.float32)[None], pts[:, 1].astype(np.float32)[None]
            g = np.abs(cv2.remap(GX, mx, my, cv2.INTER_LINEAR)[0] * n[0] + cv2.remap(GY, mx, my, cv2.INTER_LINEAR)[0] * n[1])
            j = int(g.argmax())
            if j == 0 or j == len(g) - 1 or g[j] < 200:
                continue
            den = g[j-1] - 2 * g[j] + g[j+1]
            d = 0.5 * (g[j-1] - g[j+1]) / den * 0.25 if den else 0.0
            p = pts[j] + d * n
            rows.append([n[0], n[1], 1.0]); rhs.append(n @ p); sides.append(k)
    rows, rhs, sides = np.array(rows), np.array(rhs), np.array(sides)
    if len(rows) < 60 or len(set(sides.tolist())) < 5:
        return None
    keep = np.ones(len(rows), bool)
    for _ in range(3):
        sol, *_ = np.linalg.lstsq(rows[keep], rhs[keep], rcond=None)
        res = rows @ sol - rhs
        sd = 1.4826 * np.median(np.abs(res[keep])) + 1e-3
        keep = np.abs(res) < max(3 * sd, 0.3)
    if len(set(sides[keep].tolist())) < 5:
        return None
    return float(sol[0]), float(sol[1]), float(sol[2]), float(np.sqrt(np.mean(res[keep] ** 2))), int(keep.sum())


def find_near(bgr, centre, win=40, region=520, a_range=(35, 110)):
    """Search only near an expected centre (render px). Used when the open search
    fails, e.g. the set-10 parchment frames whose gold-on-tan hexagon is too faint
    to win against the artwork. In a real photo the card is straightened first, so
    the centre is known to within the cut-edge error; a 40px (1.7mm) window is ample."""
    roi = bgr[:region, :region]
    GX, GY = grad_img(roi)
    fx = np.arange(centre[0] - win, centre[0] + win + 0.01, 1.0)[None, :, None]
    fy = np.arange(centre[1] - win, centre[1] + win + 0.01, 1.0)[:, None, None]
    return _peaks_and_refine(GX, GY, fx, fy, a_range)


def _peaks_and_refine(GX, GY, fx, fy, a_range):
    fa = np.arange(a_range[0], a_range[1], 0.5)[None, None, :]
    m = _score(GX, GY, fx, fy, fa)
    jy, jx = np.unravel_index(m.max(2).argmax(), m.shape[:2])
    prof = m[jy, jx]
    peaks = [i for i in range(2, len(prof) - 2)
             if prof[i] == prof[i-2:i+3].max() and prof[i] > 0.35 * prof.max()]
    peaks = sorted(peaks, key=lambda i: -prof[i])[:4]
    out = []
    for i in peaks:
        r = refine(GX, GY, fx[0, jx, 0], fy[jy, 0, 0], fa[0, 0, i])
        if r:
            out.append(r + (float(prof[i]),))
    return sorted(out, key=lambda b: b[2])


def find(bgr, region=520, a_range=(35, 110)):
    roi = bgr[:region, :region]; sc = 0.5
    small = cv2.resize(roi, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
    gx, gy = grad_img(cv2.GaussianBlur(small, (3, 3), 0))
    cxs = np.arange(30, region * sc - 30, 2.0)[None, :, None]
    cys = np.arange(30, region * sc - 30, 2.0)[:, None, None]
    As = np.arange(a_range[0] * sc, a_range[1] * sc, 1.0)[None, None, :]
    best = _score(gx, gy, cxs, cys, As).max(2)
    iy, ix = np.unravel_index(best.argmax(), best.shape)
    cx0, cy0 = cxs[0, ix, 0] / sc, cys[iy, 0, 0] / sc
    GX, GY = grad_img(roi)
    fx = np.arange(cx0 - 4, cx0 + 4.01, 0.5)[None, :, None]
    fy = np.arange(cy0 - 4, cy0 + 4.01, 0.5)[:, None, None]
    return _peaks_and_refine(GX, GY, fx, fy, a_range)


FIELDS = ["id", "w", "h", "search", "n_bounds", "cx", "cy", "apothems", "fit_rms", "strengths", "cx_each", "cy_each"]


# Known boundary sets (render px apothems). A result matching none of them is
# re-run with the windowed search around the standard centre.
KNOWN = [(57.74, 66.62), (67.72, 78.56), (75.69, 87.68)]
STD_CENTRE = (146.5, 156.3)


def matches_known(b):
    ap = [x[2] for x in b]
    return any(all(any(abs(a - k) < 1.2 for a in ap) for k in ks) for ks in KNOWN)


def work(path):
    cid = os.path.splitext(os.path.basename(path))[0]
    im = cv2.imread(path)
    if im is None:
        return {"id": cid, "n_bounds": -1}
    h, w = im.shape[:2]
    if w > h:                       # landscape render: survey it as printed (cost top-left when rotated)
        im = cv2.rotate(im, cv2.ROTATE_90_CLOCKWISE)
    b, how = find(im), "open"
    if not matches_known(b):
        b2 = find_near(im, STD_CENTRE)
        if matches_known(b2):
            b, how = b2, "window"
    row = {"id": cid, "w": w, "h": h, "search": how, "n_bounds": len(b)}
    if b:
        wts = np.array([x[5] for x in b])
        row.update(cx=f"{np.average([x[0] for x in b], weights=wts):.3f}",
                   cy=f"{np.average([x[1] for x in b], weights=wts):.3f}",
                   apothems=";".join(f"{x[2]:.3f}" for x in b),
                   fit_rms=";".join(f"{x[3]:.3f}" for x in b),
                   strengths=";".join(f"{x[5]:.0f}" for x in b),
                   cx_each=";".join(f"{x[0]:.3f}" for x in b),
                   cy_each=";".join(f"{x[1]:.3f}" for x in b))
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--redo-unmatched", action="store_true",
                    help="re-run rows that match no known hexagon and rewrite the CSV")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    a = ap.parse_args()
    if a.redo_unmatched:
        rows = list(csv.DictReader(open(OUT, newline="")))
        def unmatched(r):
            if int(r["n_bounds"]) <= 0: return True
            ap = [(0, 0, float(x)) for x in r["apothems"].split(";")]
            return not matches_known(ap)
        redo = [r["id"] for r in rows if unmatched(r)]
        with Pool(a.workers) as pool:
            new = {r["id"]: r for r in pool.map(work, [os.path.join(DB, "images", i + ".jpg") for i in redo])}
        with open(OUT, "w", newline="") as f:
            wr = csv.DictWriter(f, FIELDS, lineterminator="\n", restval=""); wr.writeheader()
            for r in rows:
                wr.writerow(new.get(r["id"], {k: r.get(k) or ("open" if k == "search" else "") for k in FIELDS}))
        print(f"redid {len(redo)}; still unmatched: {sum(unmatched(r) for r in new.values())}")
        sys.exit(0)
    done = set()
    if os.path.exists(OUT):
        done = {r["id"] for r in csv.DictReader(open(OUT, newline=""))}
    todo = [p for p in sorted(glob.glob(os.path.join(DB, "images", "*.jpg")))
            if os.path.splitext(os.path.basename(p))[0] not in done]
    if a.limit:
        todo = todo[:a.limit]
    new = not os.path.exists(OUT)
    t0 = time.time()
    with open(OUT, "a", newline="") as f, Pool(a.workers) as pool:
        wr = csv.DictWriter(f, FIELDS, lineterminator="\n")
        if new:
            wr.writeheader()
        for row in pool.imap(work, todo, chunksize=4):
            wr.writerow(row); f.flush()
    print(f"surveyed {len(todo)} in {time.time()-t0:.1f}s; {len(done)+len(todo)} total")
