"""Regression: the back's gold frame line must not depend on how densely it
is sampled (v0.4.0).

Before v0.4.0 the frame line was traced on the same 55 lines as the cut
edges, and on the TAG copy Y back the L/R ratio read 52.89 at 55 lines
against 51.16 / 50.98 / 51.15 at 101 / 151 / 201 - reproducible, and
wrong. It is now traced on its own fixed-pitch grid
(back.FRAME_SCAN_PITCH_MM), plus a coarser non-overlapping grid whose
difference is carried in the statistical term.

This pins, on the phone-resolution white-paper Gadget back, that neither
the cut edges' line count nor the frame pitch (inside the settled range)
moves the answer by more than a fraction of its own uncertainty. The TAG
scans, where the effect was found, are covered by test_tag_scans.py.
"""
from pathlib import Path

import pytest

import centering.back as BK
from centering.games.lorcana import LORCANA

BACK = Path(__file__).resolve().parents[2] / "fixtures" / "IMG_6398.HEIC"

pytestmark = pytest.mark.skipif(not BACK.exists(),
                                reason="Gadget white back not present")


def _ratios(monkeypatch, pitch=None, n_scans=55):
    if pitch is not None:
        monkeypatch.setattr(BK, "FRAME_SCAN_PITCH_MM", pitch)
    r = BK.analyze_back(BACK, LORCANA, make_overlay=False, n_scans=n_scans)
    return r.ratio_lr, r.ratio_tb


def test_cut_edge_line_count_does_not_move_the_ratio(monkeypatch):
    lr55, tb55 = _ratios(monkeypatch)
    lr101, tb101 = _ratios(monkeypatch, n_scans=101)
    assert abs(lr55.first_pct - lr101.first_pct) < 0.3
    assert abs(tb55.first_pct - tb101.first_pct) < 0.3


def test_frame_pitch_inside_settled_range(monkeypatch):
    lr4, tb4 = _ratios(monkeypatch, pitch=0.4)
    lr6, tb6 = _ratios(monkeypatch, pitch=0.6)
    for a, b in ((lr4, lr6), (tb4, tb6)):
        assert abs(a.first_pct - b.first_pct) < 0.5 * a.uncertainty_pts.total
