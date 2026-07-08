#!/usr/bin/env python3
"""Incrementally sync card_db/ with the current lorcanajson allCards.json.

Downloads renders only for cards that are new (or whose local image is
missing/corrupt), rewrites index.json, and appends the new signatures to
sig_index.json so auto-detect can see the new cards. Safe to re-run;
near-instant when nothing changed. As a side effect it force-refreshes the
analyzer's cached allCards.json (the stale-cache failure of 2026-07-08).

Run:  python card_db/update_db.py
"""
from __future__ import annotations

import datetime
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BASE = Path(__file__).resolve().parent          # card_db/
REPO = BASE.parent
sys.path.insert(0, str(REPO / "src"))

import cv2  # noqa: E402

from centering.cache import DiskCache  # noqa: E402
from centering.games.lorcana import ALLCARDS_ZIP_URL  # noqa: E402
from centering.identify import _color_sig, _phash  # noqa: E402

IMAGES_DIR = BASE / "images"
INDEX_FILE = BASE / "index.json"
SIG_FILE = BASE / "sig_index.json"

USER_AGENT = "centering-analyzer/0.1"
THREADS = 8
TIMEOUT = 60
RETRIES = 3
MIN_SIZE = 10000


def _fsync_write(path: Path, data: bytes):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def sanitize(s):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(s))


def variant_suffix(card):
    """Distinguishing letter for same-number variants (fetch_images parity)."""
    number = str(card.get("number"))
    token = (card.get("fullIdentifier") or "").split("\u2022")[0]
    token = token.split("/")[0].strip()
    if token and token != number and token.startswith(number):
        suffix = token[len(number):]
        if suffix:
            return sanitize(suffix)
    return None


def build_manifest(cards):
    manifest, seen = [], {}
    for c in cards:
        card_id = f"{sanitize(c.get('setCode'))}-{sanitize(c.get('number'))}"
        promo = c.get("promoGrouping")
        if promo:
            card_id += f"-{sanitize(promo)}"
        suffix = variant_suffix(c)
        if suffix:
            card_id += f"-{suffix}"
        if card_id in seen:
            print(f"WARNING duplicate id {card_id}; disambiguating with "
                  f"internal id {c.get('id')}")
            card_id = f"{card_id}-{c.get('id')}"
        seen[card_id] = c.get("id")
        manifest.append({
            "id": card_id,
            "file": f"images/{card_id}.jpg",
            "name": c.get("name"),
            "version": c.get("version"),
            "fullIdentifier": c.get("fullIdentifier"),
            "setCode": c.get("setCode"),
            "number": c.get("number"),
            "promoGrouping": promo,
            "rarity": c.get("rarity"),
            "url": (c.get("images") or {}).get("full"),
        })
    return manifest


def download_one(entry):
    fp = IMAGES_DIR / f"{entry['id']}.jpg"
    if not entry["url"]:
        return entry, "no url"
    last_err = None
    for attempt in range(1, RETRIES + 1):
        try:
            req = urllib.request.Request(entry["url"],
                                         headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                data = resp.read()
            if len(data) <= MIN_SIZE:
                last_err = f"too small ({len(data)} bytes)"
                time.sleep(2 ** attempt)
                continue
            _fsync_write(fp, data)
            return entry, None
        except urllib.error.HTTPError as e:
            last_err = f"HTTPError {e.code}"
            if e.code == 404:
                break
            time.sleep(2 ** attempt)
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(2 ** attempt)
    return entry, last_err


def image_ok(entry):
    """(width, height, bytes) if the local image decodes, else None."""
    fp = IMAGES_DIR / f"{entry['id']}.jpg"
    try:
        size = fp.stat().st_size
    except OSError:
        return None
    if size <= MIN_SIZE:
        return None
    img = cv2.imread(str(fp))
    if img is None:
        return None
    h, w = img.shape[:2]
    return w, h, size


def signature_for(entry):
    fp = IMAGES_DIR / f"{entry['id']}.jpg"
    bgr = cv2.imread(str(fp), cv2.IMREAD_REDUCED_COLOR_4)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return {"id": entry["id"],
            "card_id": (f"{entry['promoGrouping']}-{entry['number']}"
                        if entry["promoGrouping"]
                        else f"{entry['setCode']}-{entry['number']}"),
            "display": entry.get("fullIdentifier") or entry.get("name"),
            "name": entry.get("name"), "version": entry.get("version"),
            "number": entry.get("number"), "setCode": entry.get("setCode"),
            "promoGrouping": entry.get("promoGrouping"),
            "url": entry.get("url"), "file": entry.get("file"),
            "phash": _phash(cv2.convertScaleAbs(gray)),
            "hist": _color_sig(rgb).tolist()}


def main():
    IMAGES_DIR.mkdir(exist_ok=True)
    # Force-fresh allCards (also repairs a stale analyzer cache).
    cards = DiskCache().fetch_json_maybe_zipped(ALLCARDS_ZIP_URL,
                                                ttl_days=0)["cards"]
    manifest = build_manifest(cards)
    old = {}
    if INDEX_FILE.exists():
        old = {e["id"]: e
               for e in json.loads(INDEX_FILE.read_text(encoding="utf-8"))
               .get("cards", [])}
    need = [e for e in manifest
            if e["id"] not in old
            or not (IMAGES_DIR / f"{e['id']}.jpg").exists()]
    print(f"upstream {len(manifest)} cards; local index {len(old)}; "
          f"to fetch {len(need)}")

    failures = {}
    if need:
        with ThreadPoolExecutor(max_workers=THREADS) as pool:
            futs = {pool.submit(download_one, e): e for e in need}
            for fut in as_completed(futs):
                entry, err = fut.result()
                if err:
                    failures[entry["id"]] = err
                    print(f"FAIL {entry['id']}: {err}")
                else:
                    print(f"fetched {entry['id']} ({entry['fullIdentifier']})")

    # Rebuild index: carry over dimensions for untouched entries, measure new.
    out_cards, skipped = [], []
    for e in manifest:
        prev = old.get(e["id"])
        if (prev and prev.get("width")
                and (IMAGES_DIR / f"{e['id']}.jpg").exists()
                and e["id"] not in {x["id"] for x in need}):
            e["width"], e["height"], e["bytes"] = (
                prev["width"], prev["height"], prev["bytes"])
        else:
            ok = image_ok(e)
            if not ok:
                skipped.append(e["id"])
                continue
            e["width"], e["height"], e["bytes"] = ok
        out_cards.append(e)
    out_cards.sort(key=lambda e: e["id"])   # match original index ordering
    payload = {"generated": datetime.datetime.now(datetime.timezone.utc)
               .strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
               "source": "lorcanajson.org / api.lorcana.ravensburger.com",
               "cards": out_cards}
    _fsync_write(INDEX_FILE,
                 json.dumps(payload, indent=2).encode("utf-8"))
    print(f"index.json: {len(out_cards)} cards"
          + (f" (skipped, no usable image: {skipped})" if skipped else ""))

    # Incremental signature index for auto-detect.
    sig_entries = []
    if SIG_FILE.exists():
        sig_entries = json.loads(SIG_FILE.read_text(encoding="utf-8"))["entries"]
    have = {s["id"] for s in sig_entries}
    valid = {e["id"] for e in out_cards}
    sig_entries = [s for s in sig_entries if s["id"] in valid]
    added = 0
    for e in out_cards:
        if e["id"] in have:
            continue
        sig = signature_for(e)
        if sig:
            sig_entries.append(sig)
            added += 1
    _fsync_write(SIG_FILE, json.dumps(
        {"count": len(sig_entries), "hash_bits": 64,
         "entries": sig_entries}).encode("utf-8"))
    print(f"sig_index.json: {len(sig_entries)} signatures ({added} added)")
    if failures:
        print(f"{len(failures)} downloads failed; re-run to retry.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
