"""Fourth-edge wiring in analyze_borderless, end to end on a synthetic
photo: a textured card on white paper whose bottom cut is buried in a
band of high-contrast noise, matched against a fake 'official picture'.

Checks the bookkeeping, not accuracy: the constructed edge is reported as
inferred/estimated, the left-to-right result is still measured, the
top-to-bottom result is downgraded to an estimate and carries the
inferred edge's uncertainty."""
import dataclasses

import numpy as np
import pytest
from PIL import Image

from centering import analyze_borderless
from centering.games.lorcana import LORCANA

# the fake render IS the card, so the real render's crop and span
# calibration do not apply
GAME = dataclasses.replace(LORCANA, render_span_bounds_mm=None,
                           render_crop_bias_mm=None)
PPM = 20.0
RW, RH = int(round(LORCANA.card_w_mm * PPM)), int(round(LORCANA.card_h_mm * PPM))


def _texture(h, w, seed):
    rng = np.random.default_rng(seed)
    small = rng.uniform(40, 150, (h // 8 + 1, w // 8 + 1)).astype(np.uint8)
    t = Image.fromarray(small).resize((w, h), Image.BICUBIC)
    return np.asarray(t, np.float32)


class FakeRender:
    def __init__(self, render):
        self.render = render

    def get_render(self, card_id):
        rgb = np.repeat(self.render[..., None], 3, axis=2).astype(np.uint8)
        return self.render, rgb, "synthetic://render", {"fullIdentifier": card_id}


@pytest.fixture(scope="module")
def scene(tmp_path_factory):
    art = _texture(RH, RW, 1)
    H, W = 2600, 1900
    x0, y0 = 320, 350
    photo = np.full((H, W), 235.0, np.float32)
    photo[y0:y0 + RH, x0:x0 + RW] = art
    # bury the bottom cut: +-3mm of strong noise straddling it
    rng = np.random.default_rng(7)
    yb = y0 + RH
    band = slice(yb - int(3 * PPM), yb + int(3 * PPM))
    photo[band, x0 - 40:x0 + RW + 40] = rng.choice(
        [20.0, 240.0], size=(band.stop - band.start, RW + 80))
    path = tmp_path_factory.mktemp("synth") / "buried_bottom.png"
    Image.fromarray(photo.clip(0, 255).astype(np.uint8)).save(path)
    box = (x0, y0, x0 + RW, y0 + RH)
    return path, art, box


@pytest.fixture(scope="module")
def res(scene):
    path, art, box = scene
    return analyze_borderless(path, "synthetic", GAME,
                              render_source=FakeRender(art),
                              make_overlay=False, manual_bbox=box)


def test_bottom_is_inferred(res):
    b = next(e for e in res.edge_fits if e.edge == "bottom")
    assert b.method == "inferred"
    assert b.status == "estimated"
    assert any("worked out from the top edge" in n for n in b.notes)
    assert any(q.code == "EDGE_INFERRED" for q in res.qa)


def test_axes_are_labelled_honestly(res):
    x, y = res.shift_mm["x"], res.shift_mm["y"]
    assert x.status == "measured"
    assert y.status == "estimated"
    # the inferred edge's own error (>= size spread + shape floor) is
    # inside the quoted edge-definition term
    assert y.uncertainty.edge_definition >= 0.18
    assert y.uncertainty.total < 0.5          # under the estimate cap
    assert res.equivalent_ratio_tb.status == "estimated"
    # truth is zero on both axes (render == card)
    assert abs(y.value) < 0.15
    assert abs(x.value) < 0.08

