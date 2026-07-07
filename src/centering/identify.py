"""Automatic card-ID detection from a front-of-card photo.

Two cooperating strategies, cheapest first:

1. OCR the collector number printed at the bottom-left of a Lorcana front
   (e.g. "147/204 * EN * 12" -> number 147, setCode 12). Lower power; tried
   first when a Tesseract engine is available, but only *trusted* when the
   candidate it points at is confirmed by a strong render match. Degrades
   gracefully to (2).
2. Image match against the local render database (card_db). A cheap global
   signature (perceptual hash + coarse colour histogram) prefilters the
   3,211 renders to a short candidate list.

Every candidate - from either strategy - is *self-verified* by SIFT feature
matching the photo to that card's official render (the same matcher the
analyzer uses). The correct card yields hundreds of RANSAC inliers; a wrong
card throws or yields almost none. The winner is the candidate with the most
inliers, so a guess is never returned unconfirmed and near-duplicate prints
(e.g. the two Enchanted numberings of one card) resolve to the print the
photo actually matches best. Confidence is reported; low-confidence guesses
are flagged rather than silently trusted (project ethos: never guess).

The returned card_id is a string the analyzer's resolver accepts
("<number>/<promoGrouping>" for promos, else "<setCode>:<number>").

Run as a script to (re)build the signature index:
    python -m centering.identify --out card_db/sig_index.json
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .imgio import load_photo
from .locate import card_component_bbox
from .render_match import match_to_render  # noqa: F401  (parity reference)

# Inlier thresholds for the SIFT self-verify. borderless.py sees ~600-800
# inliers full-res for a correct render match and flags <200 as weak; we
# downscale for speed, so the bars are lower. STRONG => trust; WEAK..STRONG
# => report but flag as low confidence; below WEAK is not a match.
_STRONG_INLIERS = 120
_WEAK_INLIERS = 45
_VERIFY_MAX_DIM = 1400   # downscale photo/render longest side before SIFT
_RATIO = 0.72            # match_to_render parity
_RANSAC_PX = 3.0
_N_FEATURES = 8000
_HARD_CAP = 14           # max candidates to SIFT-verify
_STOP_WINDOW = 4         # after the first strong hit, verify a few more (to
#                          catch a better-matching near-duplicate) then stop


# --------------------------------------------------------------------------
# result type
# --------------------------------------------------------------------------
@dataclass
class DetectionResult:
    card_id: Optional[str] = None      # resolver-ready id, or None
    display: str = ""                  # human label (fullIdentifier / name)
    confidence: str = "none"           # "high" | "low" | "none"
    method: str = "none"               # "ocr+verify" | "image-match" | "none"
    verified: bool = False             # SIFT-confirmed against the render
    n_inliers: int = 0
    candidates: list = field(default_factory=list)  # [{card_id, display, inliers}]
    message: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------
# global signature (perceptual hash + coarse colour histogram)
# --------------------------------------------------------------------------
def _phash(gray_u8: np.ndarray, hash_size: int = 8) -> int:
    """DCT perceptual hash -> 64-bit int (hash_size=8)."""
    img = cv2.resize(gray_u8, (hash_size * 4, hash_size * 4),
                     interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(img)
    low = dct[:hash_size, :hash_size]
    med = np.median(low[1:, 1:])  # exclude DC term from the median
    bits = (low > med).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def _color_sig(rgb: np.ndarray, bins: int = 8) -> np.ndarray:
    """Coarse normalised Hue-Saturation histogram (L1-normalised)."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [bins, bins], [0, 180, 0, 256])
    hist = hist.flatten()
    s = hist.sum()
    return (hist / s) if s > 0 else hist


def _rectify_card(rgb: np.ndarray, gray: np.ndarray):
    """Crop to the card bounding box when we can find one; else pass through.

    Detection only needs an approximate upright card crop, so a bbox is
    plenty - no need for the full sub-pixel edge fit the analyzer uses.
    """
    try:
        x0, y0, x1, y1 = card_component_bbox(gray)
        return rgb[y0:y1, x0:x1], gray[y0:y1, x0:x1]
    except Exception:
        return rgb, gray


def _signature(rgb: np.ndarray, gray: np.ndarray) -> dict:
    crop_rgb, crop_gray = _rectify_card(rgb, gray)
    g8 = cv2.convertScaleAbs(crop_gray)
    return {"phash": _phash(g8), "hist": _color_sig(crop_rgb).tolist()}


def _hamming64(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# --------------------------------------------------------------------------
# card_id construction from an index.json record
# --------------------------------------------------------------------------
def construct_card_id(rec: dict) -> Optional[str]:
    """Resolver-ready id from a card_db/index.json record.

    Promos carry a promoGrouping ("6/C2"); base cards use setCode:number
    ("12:147"), which the resolver requires to have no promoGrouping.
    """
    num = rec.get("number")
    grp = rec.get("promoGrouping")
    sc = rec.get("setCode")
    if grp:
        return f"{num}/{grp}"
    if sc is not None and num is not None:
        return f"{sc}:{num}"
    return None


# --------------------------------------------------------------------------
# signature index over the local render DB
# --------------------------------------------------------------------------
def build_signature_index(index_json: Path, images_dir: Path,
                          out_path: Path, verbose: bool = False) -> int:
    """Precompute per-render signatures -> out_path (derived numbers only).

    Reads card_db/index.json + local render jpgs (at reduced resolution -
    fast, and plenty for a global signature) and writes a compact JSON the
    detector prefilters against without needing the 940MB image set at
    runtime; renders for the shortlist are fetched on demand for verify.
    """
    import os
    cards = json.loads(Path(index_json).read_text())["cards"]
    entries = []
    n = 0
    for rec in cards:
        fp = Path(images_dir) / Path(rec["file"]).name
        if not fp.exists():
            continue
        bgr = cv2.imread(str(fp), cv2.IMREAD_REDUCED_COLOR_4)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        entries.append({
            "id": rec.get("id"),
            "card_id": construct_card_id(rec),
            "display": rec.get("fullIdentifier") or rec.get("name"),
            "name": rec.get("name"), "version": rec.get("version"),
            "number": rec.get("number"), "setCode": rec.get("setCode"),
            "promoGrouping": rec.get("promoGrouping"),
            "url": rec.get("url"), "file": rec.get("file"),
            "phash": _phash(cv2.convertScaleAbs(gray)),
            "hist": _color_sig(rgb).tolist(),
        })
        n += 1
        if verbose and n % 400 == 0:
            print(f"  ... {n} renders")
    payload = {"count": len(entries), "hash_bits": 64, "entries": entries}
    with open(out_path, "wb") as f:
        f.write(json.dumps(payload).encode("utf-8"))
        f.flush()
        os.fsync(f.fileno())
    return len(entries)


_SIG_CACHE: dict = {}


def load_signature_index(path: Path) -> Optional[dict]:
    path = Path(path)
    key = str(path)
    if key in _SIG_CACHE:
        return _SIG_CACHE[key]
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    for e in data["entries"]:
        e["_hist"] = np.asarray(e["hist"], dtype=np.float32)
    _SIG_CACHE[key] = data
    return data


def _prefilter(sig: dict, index: dict, top_k: int) -> list:
    qh = sig["phash"]
    qhist = np.asarray(sig["hist"], dtype=np.float32)
    scored = []
    for e in index["entries"]:
        d_hash = _hamming64(qh, e["phash"]) / 64.0              # 0..1
        d_hist = float(np.abs(qhist - e["_hist"]).sum()) / 2.0  # 0..1
        scored.append((0.6 * d_hash + 0.4 * d_hist, e))
    scored.sort(key=lambda t: t[0])
    return [e for _, e in scored[:top_k]]


# --------------------------------------------------------------------------
# OCR of the printed collector number
# --------------------------------------------------------------------------
def _ocr_available() -> bool:
    try:
        import pytesseract  # noqa: F401
        import shutil
        return shutil.which("tesseract") is not None
    except Exception:
        return False


def ocr_collector_number(rgb: np.ndarray, gray: np.ndarray) -> dict:
    """Read the bottom-left collector line. Returns {number, setCode, raw}.

    Lorcana prints e.g. "147/204 * EN * 12": number/set-size then setCode.
    Best-effort: any field may be None if OCR is unavailable or unreadable,
    in which case detection falls back to the image match.
    """
    out = {"number": None, "setCode": None, "raw": ""}
    if not _ocr_available():
        return out
    import pytesseract
    _, crop_gray = _rectify_card(rgb, gray)
    h, w = crop_gray.shape
    y0, y1 = int(h * 0.90), int(h * 0.995)      # bottom-left collector line
    x0, x1 = int(w * 0.02), int(w * 0.55)
    region = crop_gray[y0:y1, x0:x1]
    if region.size == 0:
        return out
    up = cv2.resize(region, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    up8 = cv2.convertScaleAbs(up)
    variants = [up8, cv2.bitwise_not(up8),
                cv2.threshold(up8, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
                cv2.threshold(cv2.bitwise_not(up8), 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]]
    cfg = "--psm 7 -c tessedit_char_whitelist=0123456789/ENP "
    texts = []
    for v in variants:
        try:
            t = pytesseract.image_to_string(v, config=cfg).strip()
        except Exception:
            continue
        if t:
            texts.append(t)
    raw = " | ".join(texts)
    out["raw"] = raw
    out["number"], out["setCode"] = parse_collector_text(raw)
    return out


def parse_collector_text(raw: str):
    """Pull (number, setCode) out of an OCR'd collector line.

    Lorcana prints "<number>/<set-size> * EN * <setCode>", e.g.
    "147/204 * EN * 12" -> (147, "12"). Either field may be None.
    """
    number = setCode = None
    m = re.search(r"(\d{1,3})\s*/\s*(\d{2,3})", raw)   # "147/204"
    if m:
        number = int(m.group(1))
    ms = re.findall(r"(?:EN|E N)\D{0,4}(\d{1,2})", raw)  # setCode after EN
    if ms:
        setCode = ms[-1]
    return number, setCode


# --------------------------------------------------------------------------
# SIFT self-verification (photo features computed once, reused per render)
# --------------------------------------------------------------------------
def _downscale(gray: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = gray.shape[:2]
    m = max(h, w)
    if m <= max_dim:
        return gray
    s = max_dim / m
    return cv2.resize(gray, (int(w * s), int(h * s)),
                      interpolation=cv2.INTER_AREA)


def _photo_features(gray: np.ndarray):
    """SIFT keypoints/descriptors for the photo, computed once."""
    p8 = cv2.convertScaleAbs(_downscale(gray, _VERIFY_MAX_DIM))
    sift = cv2.SIFT_create(nfeatures=_N_FEATURES)
    kp, des = sift.detectAndCompute(p8, None)
    return kp, des, sift


def _match_precomputed(kp1, des1, sift, render_gray: np.ndarray):
    """(n_inliers, reproj_px) for photo (precomputed) vs a render.

    Mirrors render_match.match_to_render (ratio 0.72, RANSAC 3.0, same seed)
    but reuses the photo's descriptors across every candidate. Returns
    (0, inf) when the card does not match rather than raising.
    """
    if des1 is None or len(kp1) < 50:
        return 0, float("inf")
    r8 = cv2.convertScaleAbs(_downscale(render_gray, _VERIFY_MAX_DIM))
    kp2, des2 = sift.detectAndCompute(r8, None)
    if des2 is None or len(kp2) < 50:
        return 0, float("inf")
    knn = cv2.BFMatcher(cv2.NORM_L2).knnMatch(des1, des2, k=2)
    good = [m for m, n in knn if m.distance < _RATIO * n.distance]
    if len(good) < 30:
        return 0, float("inf")
    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    cv2.setRNGSeed(20260703)
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, _RANSAC_PX)
    if H is None or inl is None or int(inl.sum()) < 25:
        return 0, float("inf")
    inl = inl.ravel().astype(bool)
    proj = cv2.perspectiveTransform(src[inl], H).reshape(-1, 2)
    err = np.linalg.norm(proj - dst[inl].reshape(-1, 2), axis=1)
    return int(inl.sum()), float(np.median(err))


def _render_gray_for(rec: dict, images_dir: Optional[Path], cache) -> Optional[np.ndarray]:
    """Local render if present, else fetch by url via the DiskCache."""
    if images_dir is not None and rec.get("file"):
        fp = Path(images_dir) / Path(rec["file"]).name
        if fp.exists():
            g = cv2.imread(str(fp), cv2.IMREAD_GRAYSCALE)
            if g is not None:
                return g.astype(np.float32)
    url = rec.get("url")
    if url and cache is not None:
        try:
            p = cache.fetch(url, suffix=".img")
            buf = np.frombuffer(Path(p).read_bytes(), np.uint8)
            bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if bgr is not None:
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
        except Exception:
            return None
    return None


# --------------------------------------------------------------------------
# top-level detection
# --------------------------------------------------------------------------
def _verify_candidates(cands, kp1, des1, sift, images_dir, cache):
    """SIFT-verify candidate records; return hits [(inliers, reproj, rec)].

    Stops after the hard cap, or a short window past the first strong hit
    (enough to catch a better-matching near-duplicate that sorts adjacent).
    """
    hits = []
    scanned = 0
    first_strong_at = None
    for rec in cands:
        rg = _render_gray_for(rec, images_dir, cache)
        if rg is None:
            continue
        inl, med = _match_precomputed(kp1, des1, sift, rg)
        scanned += 1
        if inl >= _WEAK_INLIERS:
            hits.append((inl, med, rec))
        if inl >= _STRONG_INLIERS and first_strong_at is None:
            first_strong_at = scanned
        if first_strong_at is not None and scanned >= first_strong_at + _STOP_WINDOW:
            break
        if scanned >= _HARD_CAP:
            break
    hits.sort(key=lambda t: -t[0])
    return hits


def detect_card_id(front_photo, index_json: Path, images_dir: Optional[Path],
                   sig_index_path: Path, cache=None, prefer_ocr: bool = True,
                   top_k: int = 24) -> DetectionResult:
    rgb, gray, _ = load_photo(front_photo)
    cards = json.loads(Path(index_json).read_text())["cards"]
    kp1, des1, sift = _photo_features(gray)

    def cand_list(hits):
        return [{"card_id": construct_card_id(r) if "file" in r else r.get("card_id"),
                 "display": r.get("fullIdentifier") or r.get("display") or r.get("name"),
                 "inliers": int(i)} for i, _, r in hits[:3]]

    def result_from(hits, method, msg):
        inl, med, rec = hits[0]
        cid = construct_card_id(rec) if "file" in rec else rec.get("card_id")
        return DetectionResult(
            card_id=cid,
            display=rec.get("fullIdentifier") or rec.get("display") or rec.get("name"),
            confidence="high" if inl >= _STRONG_INLIERS else "low",
            method=method, verified=True, n_inliers=int(inl),
            candidates=cand_list(hits), message=msg)

    # ---- strategy 1 (cheap): OCR -> a few candidates -> verify ----
    ocr = ocr_collector_number(rgb, gray) if prefer_ocr else {"number": None}
    if ocr.get("number") is not None:
        num, sc = ocr["number"], ocr.get("setCode")
        cand = ([c for c in cards if c.get("number") == num
                 and str(c.get("setCode")) == str(sc)] if sc else [])
        if not cand:
            cand = [c for c in cards if c.get("number") == num]
        hits = _verify_candidates(cand[:_HARD_CAP], kp1, des1, sift,
                                  images_dir, cache)
        # trust OCR outright only if it produced a *strong* confirmation;
        # otherwise fall through so the image match can arbitrate.
        if hits and hits[0][0] >= _STRONG_INLIERS:
            return result_from(
                hits, "ocr+verify",
                f"OCR read #{num}" + (f" set {sc}" if sc else "")
                + f"; confirmed by render match ({hits[0][0]} inliers).")

    # ---- strategy 2: image match against the render DB ----
    index = load_signature_index(sig_index_path)
    if index is None:
        return DetectionResult(
            method="none", message="Could not auto-detect the card. Enter the "
            "card ID manually (e.g. 6/C2). (No signature index found - run "
            "`python -m centering.identify` to build one.)")
    shortlist = _prefilter(_signature(rgb, gray), index, top_k)
    hits = _verify_candidates(shortlist, kp1, des1, sift, images_dir, cache)
    if hits:
        return result_from(hits, "image-match",
                           f"Matched to the render database "
                           f"({hits[0][0]} inliers).")

    return DetectionResult(
        method="none",
        message="Could not confidently identify the card. Enter the card ID "
                "manually (e.g. 6/C2).")


if __name__ == "__main__":  # build the signature index
    import argparse
    here = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="Build the render signature index.")
    ap.add_argument("--index", default=str(here / "card_db" / "index.json"))
    ap.add_argument("--images", default=str(here / "card_db" / "images"))
    ap.add_argument("--out", default=str(here / "card_db" / "sig_index.json"))
    a = ap.parse_args()
    print(f"Building signature index from {a.images} ...")
    n = build_signature_index(Path(a.index), Path(a.images), Path(a.out),
                              verbose=True)
    print(f"Wrote {n} signatures -> {a.out}")
