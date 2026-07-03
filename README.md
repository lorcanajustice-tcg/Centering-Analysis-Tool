# Centering Analyzer

Measures trading-card cut/print centering from ordinary phone photos to
~±0.1mm / ±1-2 ratio-point accuracy. Built for Disney Lorcana, extensible to
other TCGs via the `games/` adapter layer.

## Quick start (Windows)

Double-click **`run_analyzer.bat`**. The first run installs dependencies
(needs Python 3 from python.org with "Add to PATH" ticked); then your
browser opens a local page where you pick photos, choose Back / Front /
Both, enter the card ID for fronts (e.g. `6/C2`), and hit Analyze.
Results and full-resolution overlay images are saved under `results/`.
Nothing leaves your machine except fetching the official card render.

## Install (command line)

    pip install -e .          # Python 3.10+ (3.12 target)

Dependencies (installed automatically): opencv, numpy, pillow, pillow-heif.

## Usage

    centering back  photo.heic                      # bordered face (frame line)
    centering front photo.heic --card 6/C2          # borderless face (render match)
    centering card --back b.heic --front f.heic --card 6/C2   # combined + registration
    # options: --game lorcana  --out overlays_dir  --json result.json

Card IDs: `6/C2` (number/promo grouping), `7:69` (setCode:number), or a
unique name substring (`"Elsa - Ice Maker"`).

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
  mirrored-ratio consistency.
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

Dark, matte, uniformly lit background on all four sides; telephoto lens;
card centred, filling 60-80% of the frame, never touching the frame edges;
unsleeved; small tilt (<3°) is fine; avoid glare bands crossing the card
edges; flatten foil curl.

If a shot has problems, the report says so via QA flags (e.g.
`BACKGROUND_NONUNIFORM`, `INSUFFICIENT_MARGIN`, `RADIAL_DISTORTION_RISK`,
`CURL_SUSPECTED`, `WEAK_RENDER_MATCH`). One caution flags can't catch:
a perfect-fit sleeve edge can masquerade as the card edge (±0.3mm) -
always shoot unsleeved.

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
