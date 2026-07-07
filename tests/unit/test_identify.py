"""Unit tests for card-ID auto-detection helpers (no fixtures/network)."""
from pathlib import Path
import numpy as np, cv2, pytest
from centering import identify as I


def test_construct_card_id_promo():
    assert I.construct_card_id({"number": 6, "promoGrouping": "C2", "setCode": "7"}) == "C2-6"

def test_construct_card_id_base():
    assert I.construct_card_id({"number": 147, "promoGrouping": None, "setCode": "12"}) == "12-147"

def test_construct_card_id_missing():
    assert I.construct_card_id({"number": None, "promoGrouping": None, "setCode": None}) is None

@pytest.mark.parametrize("raw,num,sc", [
    ("147/204 EN 12", 147, "12"),
    ("6 / 204  * EN * 7", 6, "7"),
    ("garbage no digits", None, None),
    ("204N /Eee 1", None, None),   # garbled "EN" not matched -> no false setCode
    ("2/C2 EN 7", None, "7"),            # C2 is not a /<digits> set-size -> number None
])
def test_parse_collector_text(raw, num, sc):
    assert I.parse_collector_text(raw) == (num, sc)

def test_load_signature_index_missing(tmp_path):
    assert I.load_signature_index(tmp_path / "nope.json") is None

def test_phash_stable_and_64bit():
    g = (np.random.RandomState(0).rand(400, 300) * 255).astype(np.uint8)
    h = I._phash(g)
    assert 0 <= h < (1 << 64)
    assert I._phash(g) == h                      # deterministic
    assert I._hamming64(h, h) == 0

def test_detect_graceful_without_index_or_ocr(tmp_path):
    # a blank synthetic image: no card, no signature index, OCR disabled ->
    # honest "none" rather than a guess.
    img = np.full((600, 400, 3), 30, np.uint8)
    p = tmp_path / "blank.png"; cv2.imwrite(str(p), img)
    idx = tmp_path / "index.json"; idx.write_text('{"cards": []}')
    r = I.detect_card_id(p, idx, None, tmp_path / "no_sig.json",
                         cache=None, prefer_ocr=False)
    assert r.card_id is None and r.method == "none" and r.confidence == "none"
