"""Card-id resolution (LorcanaRenderSource.resolve) against a stub card list.

Offline: no network, no cache. Regression coverage for the 2026-07-06 bug
where number/promoGrouping lookups only matched groupings starting with
"C" (C1/C2), so promo ids like "54/P3" (and P1-P4, D23, PD1) fell through
to name matching and raised CardNotFound.
"""

import pytest

from centering.games.lorcana import CardNotFound, LorcanaRenderSource


def _card(number, grouping, set_code, name, version=None):
    return {
        "number": number,
        "promoGrouping": grouping,
        "setCode": set_code,
        "name": name,
        "version": version,
        "fullIdentifier": f"{number}/{grouping or 'NNN'} - EN - {set_code}",
    }


CARDS = [
    _card(54, "P3", "12", "Woody", "Jungle Guide"),
    _card(6, "C2", "7", "Elsa", "Ice Maker"),
    _card(69, None, "7", "Elsa", "Ice Maker"),
    _card(1, "PD1", "11", "Beast", "Snowfield Troublemaker"),
    _card(23, "D23", "1", "Mickey Mouse", "Happiest Friend"),
    _card(54, None, "1", "Rafiki", "Mysterious Sage"),
]


@pytest.fixture
def src():
    s = LorcanaRenderSource.__new__(LorcanaRenderSource)
    s.cache = None  # no network/cache: _load must never be exercised past _cards
    s._cards = CARDS
    return s


@pytest.mark.parametrize("cid,name", [
    ("54/P3", "Woody"),      # the original bug
    ("6/C2", "Elsa"),
    ("1/PD1", "Beast"),
    ("23/D23", "Mickey Mouse"),
    ("54/p3", "Woody"),      # grouping is case-insensitive
    (" 54 / P3 ", "Woody"),  # whitespace tolerated
])
def test_promo_grouping_lookup(src, cid, name):
    assert src.resolve(cid)["name"] == name


def test_set_code_number_lookup(src):
    c = src.resolve("7:69")
    assert c["name"] == "Elsa" and c["promoGrouping"] is None


def test_unique_name_lookup(src):
    assert src.resolve("Jungle Guide")["number"] == 54


@pytest.mark.parametrize("cid", [
    "54/204",           # number/total is not a lookup form
    "99/P3",            # no such promo number
    "Elsa",             # ambiguous: two printings in stub
    "Stitch",           # unknown name
])
def test_unresolvable_ids_refused(src, cid):
    with pytest.raises(CardNotFound):
        src.resolve(cid)
