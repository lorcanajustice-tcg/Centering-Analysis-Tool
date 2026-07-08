"""Layered card-id lookup fallbacks (stale-cache refresh + local card_db).

Offline: the network layer is stubbed. Regression coverage for 2026-07-08,
when a 7-day-old cached allCards.json refused "10/C2" (the card was added
upstream after the cache was written) and blocked the whole analysis, even
though card_db/ held the render locally.
"""

import json

import cv2
import numpy as np
import pytest

from centering.games.lorcana import CardNotFound, LorcanaRenderSource


def _card(number, grouping, set_code, name, variant=None):
    tok = f"{number}{variant or ''}"
    return {"number": number, "promoGrouping": grouping,
            "setCode": set_code, "name": name, "version": None,
            "variant": variant,
            "fullIdentifier": f"{tok}/{grouping or 'NNN'} \u2022 EN \u2022 {set_code}"}


STALE = [_card(6, "C2", "7", "Elsa"), _card(69, None, "7", "Elsa")]
FRESH = STALE + [_card(10, "C2", "1", "Let it Go")]


class _RefreshCache:
    """fetch_json_maybe_zipped stub that records the ttl_days it was called
    with and always serves the FRESH list."""

    def __init__(self, cards):
        self.cards = cards
        self.calls = []

    def fetch_json_maybe_zipped(self, url, ttl_days=7.0):
        self.calls.append(ttl_days)
        return {"cards": self.cards}


def _src(cards, cache=None, local_db_dir=None):
    s = LorcanaRenderSource.__new__(LorcanaRenderSource)
    s._cards = cards
    s._local_cards = None
    s._refreshed = False
    s.cache = cache
    s.local_db_dir = local_db_dir
    return s


def test_zero_match_forces_one_refresh():
    # cached list predates the card; the forced ttl_days=0 refetch finds it
    s = _src(STALE, cache=_RefreshCache(FRESH))
    assert s.resolve("10/C2")["name"] == "Let it Go"
    assert s.cache.calls == [0]


def test_refresh_happens_at_most_once_per_instance():
    s = _src(STALE, cache=_RefreshCache(STALE))  # refresh does not help
    with pytest.raises(CardNotFound):
        s.resolve("99/C2")
    with pytest.raises(CardNotFound):
        s.resolve("98/C2")
    assert s.cache.calls == [0]


def test_ambiguous_id_is_not_retried():
    class Boom:
        def fetch_json_maybe_zipped(self, url, ttl_days=7.0):
            raise AssertionError("ambiguous ids must not trigger a refetch")

    cards = STALE + [_card(24, "P2", "7", "Hiro Hamada", variant="A"),
                     _card(24, "P2", "7", "Hiro Hamada", variant="B")]
    s = _src(cards, cache=Boom())
    with pytest.raises(CardNotFound):
        s.resolve("24/P2")


def _write_local_db(tmp_path, with_image=True):
    (tmp_path / "images").mkdir()
    rec = {"id": "1-10-C2", "file": "images/1-10-C2.jpg",
           "name": "Let it Go", "version": None,
           "fullIdentifier": "10/C2 \u2022 EN \u2022 1",
           "setCode": "1", "number": 10, "promoGrouping": "C2",
           "url": "https://example.invalid/render.jpg"}
    (tmp_path / "index.json").write_text(
        json.dumps({"generated": "test", "source": "test", "cards": [rec]}),
        encoding="utf-8")
    if with_image:
        img = np.full((88, 63, 3), 128, np.uint8)
        assert cv2.imwrite(str(tmp_path / "images" / "1-10-C2.jpg"), img)
    return rec


def test_local_card_db_fallback_offline(tmp_path):
    _write_local_db(tmp_path)
    s = _src(STALE, cache=None, local_db_dir=tmp_path)  # no network at all
    c = s.resolve("10/C2")
    assert c["name"] == "Let it Go" and c["_local_file"]


def test_get_render_prefers_local_image(tmp_path):
    _write_local_db(tmp_path)
    s = _src(STALE, cache=None, local_db_dir=tmp_path)
    gray, rgb, url, card = s.get_render("C2-10")   # canonical hyphen form
    assert gray.dtype == np.float32 and rgb.shape[2] == 3
    assert url == card["_local_file"]              # no fetch attempted


def test_no_database_anywhere_raises():
    class Down:
        def fetch_json_maybe_zipped(self, url, ttl_days=7.0):
            raise OSError("offline")

    s = _src(None, cache=Down())
    with pytest.raises(CardNotFound, match="database unavailable"):
        s.resolve("10/C2")
