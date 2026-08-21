"""The TAG filename parser (calibration/tag_reference.py).

Unit-level and fixture-free: this is the dev-tooling side of the TAG
reference work, and the parser is the part that can silently misread a
filename and quietly compare against the wrong numbers.
"""
import importlib.util
import pathlib
import sys

import pytest

_PATH = (pathlib.Path(__file__).resolve().parents[2]
         / "calibration" / "tag_reference.py")
_spec = importlib.util.spec_from_file_location("tag_reference", _PATH)
tag_reference = importlib.util.module_from_spec(_spec)
# calibration/ is a script folder, not a package; register the module before
# executing it so @dataclass can resolve its own module during class creation
sys.modules[_spec.name] = tag_reference
_spec.loader.exec_module(tag_reference)

parse = tag_reference.parse_tag_filename


def test_parses_front_and_back():
    f = parse("T6453597F-47L53R44T56B.jpg")
    assert (f.serial, f.face) == ("T6453597", "front")
    assert (f.left, f.right, f.top, f.bottom) == (47, 53, 44, 56)
    assert (f.lr_pct, f.tb_pct) == (47.0, 44.0)

    b = parse("Y3106454B-55L45R49T51B.jpg")
    assert (b.serial, b.face) == ("Y3106454", "back")
    assert (b.left, b.right, b.top, b.bottom) == (55, 45, 49, 51)


def test_accepts_full_path_and_any_extension():
    assert parse(pathlib.Path("/tmp/x/T1B-50L50R50T50B.png")).face == "back"


@pytest.mark.parametrize("name", [
    "IMG_6341.HEIC",                 # ordinary photo
    "elsa_front.jpg",
    "T6453597X-47L53R44T56B.jpg",    # face letter is neither F nor B
    "T6453597F-47L53R44T56.jpg",     # missing trailing B
    "T6453597F-47L52R44T56B.jpg",    # L/R do not sum to 100
    "T6453597F-47L53R44T57B.jpg",    # T/B do not sum to 100
])
def test_rejects_non_tag_names(name):
    assert parse(name) is None


def test_percentages_must_sum_to_100_not_just_look_numeric():
    """The sum check is what keeps an arbitrary filename from being read
    as a measurement - guard it explicitly."""
    assert parse("abcF-10L20R30T40B.jpg") is None
    assert parse("abcF-10L90R30T70B.jpg") is not None
