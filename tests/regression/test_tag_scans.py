"""Regression: TAG grading-report scans of Lilo & Stitch - Fun-Loving Friends
244/207 EN (set 13, Enchanted), two graded copies, front + back.

Fixtures: fixtures/tag/{T6453597,Y3106454}{F,B}-*.jpg (git-ignored like all
card imagery). These are ~4390x6090 flatbed-grade scans cropped hard to the
card and padded with a flat orange band about 48px (0.7mm) wide - the card
fills 99% of the frame. Every side of the standard coarse pass fails on
them, because each scanner samples its background level from a window
30% of the frame wide and on this framing that window is card. They are the
reason the tight-crop localization path exists (locate.tight_crop_locate),
and they exercise three things nothing else in the suite does:

* localization with essentially no background,
* the inverted texture polarity (a flat surround against a grainy printed
  face - the opposite of the dark-mat case),
* the render match on a 27MP photo (downscaled for SIFT; see
  render_match.MATCH_MAX_LONG_EDGE - at full size it needs several GB).

The filenames carry TAG's own centering grade, which gives an INDEPENDENT
third-party measurement of the same physical cards. Bounds below are the
deltas actually observed, not a target:

    file                 TAG L/R   ours    d      TAG T/B   ours    d
    T6453597B (back)      55/45    53.3   -1.7     41/59   refused
    Y3106454B (back)      55/45    52.8   -2.1     49/51    48.9   -0.1
    T6453597F (front)     47/53    49.2   +2.2     44/56    47.0   +3.0
    Y3106454F (front)     47/53    50.0   +3.0     47/53    52.9   +5.9

The bordered-back path tracks TAG closely on T/B and sits ~2 points low on
L/R on both copies. The borderless front path (render match) is looser -
up to 6 points, ~0.25mm - which is NOT understood: it may be our render
crop bias, or TAG measuring a full-art face against something other than
the official render. Treat the front bounds as a tripwire against
regression, not as a validated accuracy claim. See DEV-NOTES.md.
"""
import importlib.util
import pathlib
import sys

import pytest

from centering.back import analyze_back
from centering.borderless import analyze_borderless
from centering.games.lorcana import LORCANA

FIXTURES = pathlib.Path(__file__).resolve().parents[2] / "fixtures" / "tag"

_PATH = (pathlib.Path(__file__).resolve().parents[2]
         / "calibration" / "tag_reference.py")
_spec = importlib.util.spec_from_file_location("tag_reference", _PATH)
tag_reference = importlib.util.module_from_spec(_spec)
# calibration/ is a script folder, not a package; register the module before
# executing it so @dataclass can resolve its own module during class creation
sys.modules[_spec.name] = tag_reference
_spec.loader.exec_module(tag_reference)

CARD_ID = "13-244"
BACKS = ["T6453597B-55L45R41T59B.jpg", "Y3106454B-55L45R49T51B.jpg"]
FRONTS = ["T6453597F-47L53R44T56B.jpg", "Y3106454F-47L53R47T53B.jpg"]

# observed |delta| vs TAG, plus headroom; see the module docstring
BACK_TOL_PTS = 3.0
FRONT_TOL_PTS = 7.0

_have = pytest.mark.skipif(
    not all((FIXTURES / n).exists() for n in BACKS + FRONTS),
    reason="TAG scan fixtures not present (see fixtures/README.md)")

pytestmark = _have


@pytest.fixture(scope="module")
def backs():
    return {n: analyze_back(FIXTURES / n, LORCANA, make_overlay=False)
            for n in BACKS}


@pytest.fixture(scope="module")
def fronts():
    return {n: analyze_borderless(FIXTURES / n, CARD_ID, LORCANA,
                                  make_overlay=False)
            for n in FRONTS}


@pytest.mark.parametrize("name", BACKS + FRONTS)
def test_filename_carries_a_reference(name):
    assert tag_reference.parse_tag_filename(name) is not None


@pytest.mark.parametrize("name", BACKS)
def test_back_cut_edges_all_fit(backs, name):
    """The point of the tight-crop path: all four cut edges measure."""
    fits = {f.edge: f for f in backs[name].edge_fits}
    for side in ("left", "right", "top", "bottom"):
        assert fits[side].status in ("ok", "flagged"), fits[side].notes
        assert fits[side].rms_residual_px < 1.5


@pytest.mark.parametrize("name", BACKS + FRONTS)
def test_scale_and_shape_are_physical(backs, fronts, name):
    """A localization that latched onto the wrong thing would not land on
    the scans' true 67.5 px/mm AND the card's aspect ratio at once."""
    res = (backs if name in BACKS else fronts)[name]
    assert res.input.px_per_mm == pytest.approx(67.5, abs=0.4)
    assert res.aspect_ratio_measured == pytest.approx(
        LORCANA.card_h_mm / LORCANA.card_w_mm, rel=0.005)


@pytest.mark.parametrize("name", BACKS + FRONTS)
def test_tight_crop_flagged_once_not_four_warnings(backs, fronts, name):
    res = (backs if name in BACKS else fronts)[name]
    codes = [f.code for f in res.qa]
    assert "TIGHT_CROP" in codes
    assert "RADIAL_DISTORTION_RISK" not in codes


@pytest.mark.parametrize("name", BACKS)
def test_back_ratios_track_tag(backs, name):
    ref = tag_reference.parse_tag_filename(name)
    r = backs[name]
    assert r.ratio_lr.first_pct == pytest.approx(ref.lr_pct, abs=BACK_TOL_PTS)
    # T6453597B's top frame line is genuinely inconsistent (frame_peak rms
    # 4.6px) and T/B is refused there - an honest refusal, not a bound to
    # weaken. Check the axis only where it measured.
    if r.ratio_tb.first_pct is not None:
        assert r.ratio_tb.first_pct == pytest.approx(ref.tb_pct,
                                                     abs=BACK_TOL_PTS)


@pytest.mark.parametrize("name", FRONTS)
def test_front_ratios_track_tag(fronts, name):
    ref = tag_reference.parse_tag_filename(name)
    r = fronts[name]
    assert r.equivalent_ratio_lr.first_pct == pytest.approx(
        ref.lr_pct, abs=FRONT_TOL_PTS)
    assert r.equivalent_ratio_tb.first_pct == pytest.approx(
        ref.tb_pct, abs=FRONT_TOL_PTS)


def test_two_copies_of_the_same_card_agree_on_lr(backs):
    """Both graded copies are 55/45 to TAG; whatever our offset is, it must
    be the same on both - a per-image wobble would mean the detector, not a
    systematic, is doing the talking."""
    a, b = (backs[n].ratio_lr.first_pct for n in BACKS)
    assert abs(a - b) <= 1.5
