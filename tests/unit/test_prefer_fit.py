"""_prefer_fit: pin the DECISION, not the constant.

Measured over the fixture corpus on 2026-08-21 (all seven two-sided edge
decisions the bordered and borderless paths make on it):

    support ratio min / median / max = 0.244 / 0.346 / 0.630
    support branch fired 7/7, the residual branch 0/7
    in 6 of the 7, the better-supported fit had the LOOSER residual

Two things follow. First, support_tol=0.75 is not a tuned number - every
value from about 0.64 to 1.0 behaves identically on this corpus, because
no decision comes closer to the threshold than 0.63. Second, it is load
bearing anyway: without the support branch, six of seven edges would have
been taken from the thin-but-tight fit instead of the well-supported one.

So the constant has slack but the behaviour matters, and these tests pin
the behaviour. If someone retunes support_tol, these should still hold;
if they no longer do, the retune changed what the function MEANS.
"""
import pytest

from centering.fitting import _prefer_fit
from centering.geometry import FittedLine


def _line(n, rms):
    ln = FittedLine("v", 0.0, 100.0, rms=rms, n=n, u_range=(0.0, 100.0))
    return ln


def _cand(n, rms, method="texture"):
    return (_line(n, rms), object(), object(), method)


def test_none_candidate_loses():
    a, b = _cand(20, 0.5), (None, None, None, None)
    assert _prefer_fit(a, b) is a
    assert _prefer_fit(b, a) is a


def test_better_supported_beats_tighter_residual():
    """The corpus case: 9 lines at rms 0.32 vs 26 lines at rms 0.68."""
    thin, solid = _cand(9, 0.32), _cand(26, 0.68)
    assert _prefer_fit(thin, solid) is solid
    assert _prefer_fit(solid, thin) is solid


def test_comparable_support_falls_through_to_residual():
    """Above the support margin the tighter fit wins, either way round."""
    loose, tight = _cand(30, 0.90), _cand(29, 0.40)
    assert _prefer_fit(loose, tight) is tight
    assert _prefer_fit(tight, loose) is tight


def test_tie_keeps_the_primary_scanner():
    """Equal support and equal residual must not displace the caller's
    primary; a coin-flip here would make results capture-order dependent."""
    a, b = _cand(25, 0.60, "texture"), _cand(25, 0.60, "step")
    assert _prefer_fit(a, b) is a


@pytest.mark.parametrize("tol", [0.64, 0.75, 0.90, 1.0])
def test_corpus_decisions_are_insensitive_to_the_constant(tol):
    """Every two-sided decision the corpus actually makes, at four values
    of support_tol spanning the range the measurement leaves open."""
    corpus = [(9, 0.323, 26, 0.677), (17, 0.615, 27, 0.791),
              (10, 0.652, 41, 0.937), (9, 0.374, 29, 0.632),
              (15, 0.668, 26, 1.273), (10, 0.492, 38, 0.906),
              (13, 1.040, 26, 0.790)]
    for na, ra, nb, rb in corpus:
        a, b = _cand(na, ra), _cand(nb, rb)
        assert _prefer_fit(a, b, support_tol=tol) is b, (na, nb, tol)
