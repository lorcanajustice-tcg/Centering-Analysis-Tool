#!/usr/bin/env python3
"""Download every released Disney Lorcana card image listed in allCards.json."""
import json
import os
import re
import sys
import time
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import urllib.request
import urllib.error

from PIL import Image

BASE_DIR = "/sessions/wizardly-lucid-bell/carddb"
CACHE_FILE = "/sessions/wizardly-lucid-bell/cache/allCards.json"
IMAGES_DIR = os.path.join(BASE_DIR, "images")
INDEX_FILE = os.path.join(BASE_DIR, "index.json")
LOG_FILE = os.path.join(BASE_DIR, "download.log")
FAILURES_FILE = os.path.join(BASE_DIR, "failures.json")

USER_AGENT = "centering-analyzer/0.1"
THREADS = 8
TIMEOUT = 60
RETRIES = 3
MIN_SIZE = 10000

_log_lock_path = LOG_FILE


def log(msg):
    line = f"{datetime.datetime.now().isoformat()} {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def sanitize(s):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(s))


def variant_suffix(card):
    """Return a letter/word suffix distinguishing same-number variant cards.

    Some cards share (setCode, number, promoGrouping) but are genuinely
    different images (e.g. set3 #4a-#4e Dalmatian Puppy variants, or
    promo2 #24A/#24B Hiro Hamada). The lorcanajson fullIdentifier carries
    the distinguishing letter (e.g. "4a/204", "24A/P2"); extract it when
    it differs from the bare number. Falls back to None if not present.
    """
    number = str(card.get("number"))
    full_id = card.get("fullIdentifier") or ""
    # fullIdentifier looks like "<num><suffix>/<set> • EN • <setCode>"
    token = full_id.split("•")[0].strip()
    token = token.split("/")[0].strip()
    if token and token != number and token.startswith(number):
        suffix = token[len(number):]
        if suffix:
            return sanitize(suffix)
    return None


def build_manifest():
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    cards = data["cards"]
    manifest = []
    seen_ids = {}
    for c in cards:
        set_code = sanitize(c.get("setCode"))
        number = sanitize(c.get("number"))
        card_id = f"{set_code}-{number}"
        promo = c.get("promoGrouping")
        if promo:
            card_id += f"-{sanitize(promo)}"
        suffix = variant_suffix(c)
        if suffix:
            card_id += f"-{suffix}"
        if card_id in seen_ids:
            # Genuine collision even after variant-suffix disambiguation
            # (e.g. two different cards both reporting fullIdentifier
            # "26/P2"). Fall back to the internal numeric id to keep
            # every card's image distinct rather than overwriting one.
            log(f"WARNING duplicate id {card_id} (card ids {seen_ids[card_id]} and {c.get('id')}); "
                f"disambiguating with internal id")
            card_id = f"{card_id}-{c.get('id')}"
        seen_ids[card_id] = c.get("id")
        url = c.get("images", {}).get("full")
        filename = f"{card_id}.jpg"
        filepath = os.path.join(IMAGES_DIR, filename)
        manifest.append({
            "id": card_id,
            "file": f"images/{filename}",
            "filepath": filepath,
            "name": c.get("name"),
            "version": c.get("version"),
            "fullIdentifier": c.get("fullIdentifier"),
            "setCode": c.get("setCode"),
            "number": c.get("number"),
            "promoGrouping": promo,
            "rarity": c.get("rarity"),
            "url": url,
        })
    return manifest


def download_one(entry):
    """Download a single card image with retries. Returns (entry, error_or_None)."""
    filepath = entry["filepath"]
    url = entry["url"]

    if not url:
        return entry, "no url"

    if os.path.exists(filepath) and os.path.getsize(filepath) > MIN_SIZE:
        return entry, None

    tmp_path = filepath + ".tmp"
    last_err = None
    for attempt in range(1, RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                data = resp.read()
            if len(data) <= MIN_SIZE:
                last_err = f"downloaded file too small ({len(data)} bytes)"
                time.sleep(2 ** attempt)
                continue
            with open(tmp_path, "wb") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, filepath)
            return entry, None
        except urllib.error.HTTPError as e:
            last_err = f"HTTPError {e.code}"
            if e.code == 404:
                # permanent failure, no point retrying
                break
            time.sleep(2 ** attempt)
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            time.sleep(2 ** attempt)
    try:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    except OSError:
        pass
    return entry, last_err


def validate_one(entry):
    """Validate a downloaded JPEG. Returns (entry, error_or_None, width, height, size)."""
    filepath = entry["filepath"]
    try:
        size = os.path.getsize(filepath)
        if size <= MIN_SIZE:
            return entry, f"too small ({size} bytes)", None, None, size
        with Image.open(filepath) as img:
            img.load()
            width, height = img.size
        return entry, None, width, height, size
    except Exception as e:
        return entry, f"{type(e).__name__}: {e}", None, None, None


def main():
    os.makedirs(IMAGES_DIR, exist_ok=True)

    log("=== Starting download run ===")
    manifest = build_manifest()
    log(f"Manifest built: {len(manifest)} cards")

    # Phase 1: download
    failures = {}
    done = 0
    total = len(manifest)
    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        futures = {pool.submit(download_one, entry): entry for entry in manifest}
        for fut in as_completed(futures):
            entry, err = fut.result()
            done += 1
            if err:
                failures[entry["id"]] = {"url": entry["url"], "error": err}
                log(f"FAIL download {entry['id']}: {err}")
            if done % 100 == 0 or done == total:
                log(f"Download progress: {done}/{total} ({len(failures)} failed so far)")

    log(f"Download phase complete: {total - len(failures)} ok, {len(failures)} failed")

    # Phase 2: validate every file that exists
    log("Starting validation phase")
    results = []
    valid_failures = {}
    done = 0
    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        futures = {pool.submit(validate_one, entry): entry for entry in manifest}
        for fut in as_completed(futures):
            entry, err, width, height, size = fut.result()
            done += 1
            if err:
                valid_failures[entry["id"]] = {"url": entry["url"], "error": err}
                log(f"FAIL validate {entry['id']}: {err}")
            else:
                results.append({
                    "id": entry["id"],
                    "file": entry["file"],
                    "name": entry["name"],
                    "version": entry["version"],
                    "fullIdentifier": entry["fullIdentifier"],
                    "setCode": entry["setCode"],
                    "number": entry["number"],
                    "promoGrouping": entry["promoGrouping"],
                    "rarity": entry["rarity"],
                    "url": entry["url"],
                    "width": width,
                    "height": height,
                    "bytes": size,
                })
            if done % 200 == 0 or done == total:
                log(f"Validation progress: {done}/{total}")

    # merge failures from both phases
    all_failures = dict(failures)
    for k, v in valid_failures.items():
        all_failures[k] = v

    log(f"Validation phase complete: {len(results)} valid, {len(valid_failures)} invalid")

    # Write index.json
    index = {
        "generated": datetime.datetime.utcnow().isoformat() + "Z",
        "source": "lorcanajson.org / api.lorcana.ravensburger.com",
        "cards": sorted(results, key=lambda r: r["id"]),
    }
    tmp_index = INDEX_FILE + ".tmp"
    with open(tmp_index, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_index, INDEX_FILE)
    log(f"Wrote index.json with {len(results)} cards")

    # Write failures.json
    if all_failures:
        tmp_fail = FAILURES_FILE + ".tmp"
        with open(tmp_fail, "w", encoding="utf-8") as f:
            json.dump(all_failures, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_fail, FAILURES_FILE)
        log(f"Wrote failures.json with {len(all_failures)} entries")
    else:
        if os.path.exists(FAILURES_FILE):
            os.remove(FAILURES_FILE)
        log("No failures")

    log("=== Run complete ===")
    log(f"SUMMARY ok={len(results)} failed={len(all_failures)} total={total}")


if __name__ == "__main__":
    main()
