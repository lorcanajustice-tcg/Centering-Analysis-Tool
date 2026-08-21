"""Unit tests for the chromaticity edge scanner (edges.colour_scan).

Synthetic images only - no fixtures, no network. The point of each case is
a property the detector must have for the reason it exists:

- it finds the cut where the HUE changes, not where the brightness does;
- a shadow ramp on the background (which is what displaces the brightness
  scanner) must not move it;
- it must refuse, not guess, when background and card share a hue.
"""
import numpy as np

from centering import edges as E


def _card_image(w=400, h=120, edge_x=150.0, bg=(240, 130, 40), card=(30, 32, 40)):
    """Coloured background on the left of edge_x, card on the right."""
    img = np.zeros((h, w, 3), np.float32)
    xs = np.arange(w)
    # pixel xs covers [xs-0.5, xs+0.5]; `edge_x` is the true cut, so the
    # background fraction of that pixel is clip(edge_x - xs + 0.5, 0, 1)
    frac = np.clip(edge_x - xs + 0.5, 0.0, 1.0)
    for c in range(3):
        img[:, :, c] = bg[c] * frac + card[c] * (1 - frac)
    return img


def _shade(img, edge_x, depth=0.55, width=40.0):
    """Multiply the background by a smooth ramp reaching the cut: a shadow.

    Scales all three channels equally, so hue is untouched and only the
    brightness scanner should care.
    """
    h, w, _ = img.shape
    xs = np.arange(w)
    t = np.clip((xs - (edge_x - width)) / width, 0.0, 1.0)
    ramp = np.where(xs < edge_x, 1.0 - depth * t, 1.0)
    return img * ramp[None, :, None]


def _fit_x(img, side="left", approx=150.0, out=60.0, inn=60.0):
    chroma = E.chromaticity(img)
    us = np.linspace(20, img.shape[0] - 20, 40)
    u, v, diag = E.colour_scan(chroma, side, approx, us,
                               search_out_px=out, search_in_px=inn)
    return u, v, diag


def test_finds_the_hue_step_subpixel():
    for edge_x in (150.0, 150.35, 150.7):
        img = _card_image(edge_x=edge_x)
        u, v, diag = _fit_x(img, approx=150.0)
        assert diag.n_ok == diag.n_attempted
        assert abs(float(np.median(v)) - edge_x) < 0.4


def test_shadow_on_the_background_does_not_move_it():
    """The regression this scanner exists for."""
    edge_x = 150.0
    clean = _card_image(edge_x=edge_x)
    shaded = _shade(clean, edge_x)
    _, v_clean, _ = _fit_x(clean)
    _, v_shaded, d = _fit_x(shaded)
    assert d.n_ok == d.n_attempted
    assert abs(float(np.median(v_shaded)) - float(np.median(v_clean))) < 0.3
    # ... while the brightness scanner walks out into the shadow
    # ... while the brightness scanner either walks out into the shadow or
    # (its shadowed-outside-level guard doing its job) declines the lines
    gray = shaded.mean(axis=2)
    us = np.linspace(20, shaded.shape[0] - 20, 40)
    gu, gv, gd = E.step_scan(gray, "left", 150.0, us,
                             search_out_px=60.0, search_in_px=60.0)
    assert gd.n_ok == 0 or float(np.median(gv)) < edge_x - 2.0


def test_refuses_when_background_and_card_share_a_hue():
    """White paper, neutral-dark card: no colour signal, so no answer."""
    img = _card_image(bg=(245, 245, 245), card=(40, 40, 40))
    _, v, diag = _fit_x(img)
    assert diag.n_ok == 0
    assert "insufficient_chroma" in diag.reject_reasons


def test_works_from_either_side():
    img = _card_image(w=400, edge_x=150.0)
    flipped = img[:, ::-1].copy()          # card now on the LEFT, cut at 249
    _, v, diag = _fit_x(flipped, side="right", approx=249.0)
    assert diag.n_ok == diag.n_attempted
    assert abs(float(np.median(v)) - 249.0) < 0.6


def test_polarity_agnostic_bright_card_on_dark_coloured_mat():
    img = _card_image(bg=(20, 60, 120), card=(230, 220, 200))
    _, v, diag = _fit_x(img)
    assert diag.n_ok == diag.n_attempted
    assert abs(float(np.median(v)) - 150.0) < 0.5


def test_frame_peak_rescue_is_additive():
    """A peak that already clears min_peak is untouched by the rescue pass."""
    from centering.geometry import FittedLine
    h, w = 60, 300
    gray = np.full((h, w), 20.0, np.float32)
    gray[:, 120:126] = 20.0 + 80.0                 # strong line at ~122.5
    
    us = np.linspace(10, h - 10, 20)
    line = FittedLine.fit("v", us, np.full_like(us, 100.0))
    _, v_hi, d_hi = E.frame_peak_scan(gray, "left", line, us, px_per_mm=10.0,
                                      min_peak=45.0, search_mm=(0.5, 12.0))
    assert d_hi.n_ok == d_hi.n_attempted
    # a duller line that misses the absolute threshold is now rescued
    gray2 = np.full((h, w), 20.0, np.float32)
    gray2[:, 120:126] = 20.0 + 40.0
    _, v_lo, d_lo = E.frame_peak_scan(gray2, "left", line, us, px_per_mm=10.0,
                                      min_peak=45.0, search_mm=(0.5, 12.0))
    assert d_lo.n_ok == d_lo.n_attempted
    assert "min_peak_rescued" in d_lo.reject_reasons
    assert abs(float(np.median(v_lo)) - float(np.median(v_hi))) < 0.5
