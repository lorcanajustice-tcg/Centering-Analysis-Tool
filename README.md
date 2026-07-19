# Centering Analyzer

Measures trading-card cut/print centering from ordinary phone photos to
~±0.1mm / ±1-2 ratio-point accuracy. Built for Disney Lorcana, extensible to
other TCGs via the `games/` adapter layer.

## Quick start (Windows)

Double-click **`run_analyzer.bat`**. The first run installs dependencies
(needs Python 3 from python.org with "Add to PATH" ticked); then your
browser opens a local page where you pick photos, choose Back / Front /
Both, and hit Analyze. For fronts, the card ID is auto-detected from the
photo (you can override it); or type it, e.g. `8-210`.
Results and full-resolution overlay images are saved under `results/`.
Nothing leaves your machine except fetching the official card render.

## Install (command line)

    pip install -e .          # Python 3.10+ (3.12 target)

Dependencies (installed automatically): opencv, numpy, pillow, pillow-heif.

## Usage

    centering back  photo.heic                      # bordered face (frame line)
    centering front photo.heic --card 8-210         # borderless face (render match)
    centering card --back b.heic --front f.heic --card 8-210  # combined + registration
    # options: --game lorcana  --out overlays_dir  --json result.json

Card IDs (`SET-NUMBER`): `8-210` (set code - card number), `C2-6` / `P1-42`
(promo/enchanted grouping - number), or a unique name substring
(`"Elsa - Ice Maker"`). The older `6/C2` and `7:69` forms are still accepted.

Library API (all functions pure; results JSON-serializable via `.to_dict()`):

    from centering import analyze_back, analyze_borderless, analyze_card
    r = analyze_back("back.heic", LORCANA)
    r.ratio_lr.display        # "55/45"
    r.borders_mm["left"]      # Measurement(value, unit, uncertainty, status)

## What it does

- **Bordered faces** (e.g. Lorcana backs): border widths on all four sides
  from physical cut edge to printed frame line; L/R and T/B grading ratios
  plus mm values.
- **Borderless faces** (e.g. Enchanted fronts): fetches the official render
  (cached under `~/.cache/centering` or `$CENTERING_CACHE`), matches your
  photo to it, and reports the print shift in mm plus an equivalent grading
  ratio.
- **Combined report**: front-back registration cross-check and
  mirrored-ratio consistency. Note: genuine cards show front-back print
  registration scatter of about +-0.19mm (one calibrated card reached
  0.43mm), so a small registration delta is normal manufacturing
  variation, not a miscut.
- **Honest uncertainty**: every number carries an uncertainty estimate.
  Quantities that can't be measured reliably are *refused* with the reason,
  never guessed.
- **Verification overlays**: every analysis writes a JPEG with the detected
  edges, frame lines, and projected render bounds drawn on your photo, so
  you can check the measurement yourself.
- **Tilt tolerance**: perspective from a slightly tilted shot is measured
  and corrected. Lens distortion is not modelled, so keep the card away
  from the photo's frame edges (you'll be warned if it's too close).

## Getting a good photo

**Background: plain white paper is the best choice** for standard cards
(black-bordered fronts and backs) - the edge detectors handle either
polarity, and white gives the strongest, cleanest contrast. A dark matte
mat also works; mid-tone backgrounds (e.g. kraft cardboard) are supported
but give the detectors less to work with.

**Light: diffuse and even, on all four sides.** Directional lamps are the
main failure mode: they cast shadow bands along card edges that can shift
an edge detection by 0.5-0.7mm (the analyzer cross-checks for this and
flags `SHADOW_BAND_SUSPECTED`, refusing obviously broken geometry, but a
clean capture beats a caught artifact).

Also: telephoto lens; card centred, filling 60-80% of the frame, with at
least 5% clearance between every card edge and the photo frame edge (lens
distortion is not modelled near the frame); unsleeved; small tilt (<3°) is
fine; flatten foil curl.

If a shot has problems, the report says so via QA flags (e.g.
`BACKGROUND_NONUNIFORM`, `SHADOW_BAND_SUSPECTED`, `ASPECT_DEVIATION`,
`RADIAL_DISTORTION_RISK`, `CURL_SUSPECTED`, `WEAK_RENDER_MATCH`). One
caution flags can't catch: a perfect-fit sleeve edge can masquerade as the
card edge (±0.3mm) - always shoot unsleeved.

## Card imagery & licensing

The MIT license covers the code in this repository only. Card artwork,
names, and official renders are Disney/Ravensburger IP and are **not**
distributed here: the analyzer fetches the single render it needs at
runtime, the full render database is rebuilt locally with
`card_db/fetch_images.py`, and test-fixture photos are local-only (see
`fixtures/README.md`).

## Tests

    pytest tests/unit          # geometry/statistics primitives, synthetic images
    pytest tests/regression    # fixture photos vs known-good values (needs local fixtures)

## Extending to other games

Add a `GameSpec` in `src/centering/games/` (card dimensions, frame-line
expectations, equivalence margins, edge-definition calibration) and a
`RenderSource` with `get_render(card_id)`. Nothing else changes.
