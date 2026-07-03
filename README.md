# Centering Analyzer

Measures trading-card cut/print centering from ordinary phone photos to
~±0.1mm / ±1-2 ratio-point accuracy. Built for Disney Lorcana, extensible to
other TCGs via the `games/` adapter layer.

## Easiest way to run (Windows)

Double-click **`run_analyzer.bat`** (in this folder). First run
installs dependencies (needs Python 3 from python.org, "Add to PATH" ticked);
then your browser opens a local page where you pick photos, choose
Back / Front / Both, enter the card ID for fronts (e.g. `6/C2`), and hit
Analyze. Results + full-res overlays are saved under `centering-analyzer/results/`.
Nothing leaves your machine except fetching the official card render.

## Install

    pip install -e .          # Python 3.10+ (3.12 target), needs opencv, numpy, pillow, pillow-heif

## Usage

    centering back  photo.heic                      # bordered face (frame line)
    centering front photo.heic --card 6/C2          # borderless face (render match)
    centering card --back b.heic --front f.heic --card 6/C2   # combined + registration
    # options: --game lorcana  --out overlays_dir  --json result.json

Card ids: `6/C2` (number/promo grouping), `7:69` (setCode:number), or a
unique name substring (`"Elsa - Ice Maker"`).

Library API (all functions pure; results JSON-serializable via `.to_dict()`):

    from centering import analyze_back, analyze_borderless, analyze_card
    r = analyze_back("back.heic", LORCANA)
    r.ratio_lr.display        # "55/45"
    r.borders_mm["left"]      # Measurement(value, unit, uncertainty, status)

## What it does

- **Bordered faces** (e.g. Lorcana backs): border widths on all four sides
  from physical cut edge to printed gold frame line; L/R and T/B grading
  ratios plus mm values.
- **Borderless faces** (e.g. Enchanted fronts): fetches the official render
  (lorcanajson.org -> Ravensburger CDN, disk-cached under
  `~/.cache/centering` or `$CENTERING_CACHE`), SIFT-matches the photo to it
  (ratio 0.72, RANSAC 3.0), maps the physical cut into render space, reports
  the print shift in mm and an equivalent grading ratio.
- **Combined report**: front-back registration cross-check and
  mirrored-ratio consistency.
- **Uncertainty**: every number carries statistical / perspective /
  edge-definition terms separately, plus RSS total. Unmeasurable quantities
  are *refused* with the reason, never guessed.
- **Verification overlays**: every analysis writes a JPEG with the detected
  edges, frame lines, and projected render bounds drawn on the photo.
- **Tilt**: keystone percentages and, when the homography permits,
  self-calibrated focal length and pitch/yaw/total tilt in degrees. A planar
  homography fully corrects perspective; radial lens distortion is NOT
  modelled - keep card edges away from the photo frame edges (warned via QA).

## Card imagery & licensing

The MIT license covers the code in this repository only. Card artwork,
names, and official renders are Disney/Ravensburger IP and are **not**
distributed here: the analyzer fetches the single render it needs at
runtime (cached under `~/.cache/centering` or `$CENTERING_CACHE`), the
full render database is rebuilt locally with `card_db/fetch_images.py`,
and test-fixture photos are local-only (see `fixtures/README.md`).

## Method notes (hard-won, do not "simplify" away)

- Dark card on dark mat: brightness thresholding, Canny+contours and Hough
  all fail. Texture is the signal - pixel std-dev across a ~24px band; mat
  grain ~10-20, smooth card border ~2-4; edge = sustained (~18px) crossing of
  the midpoint of outer-median vs inner-8th-percentile texture. Scan-line
  gate `min_sep=6`: "twilight" lines with weaker separation carry a
  systematic outward bias from edge shadows (found on the Elsa fixture).
- Bright card on dark mat: sub-pixel 50%-threshold crossing with a plateau
  check - glare bands on the mat cross the threshold but dip back down, so
  candidates without a sustained card-level plateau are skipped and the scan
  re-searches (fixture: broad glare band above the front's top edge).
- Frame line: first bright peak (>=45 over local median) scanning INWARD
  from the detected card edge; intensity-weighted centroid over +-5px. The
  Lorcana back has inner decorative doubled lines - anchoring at the edge
  avoids them.
- Line fits: repeated-median (Siegel) init + LTS C-steps (h=0.55) + MAD
  rejection + LSQ refit. Plain LSQ and Theil-Sen both tilt under a clustered
  same-sign biased tail (the shadow case). Target residuals <=1.5px, flagged
  above, refused above 4px.
- Official renders are cropped ~0.4-0.8mm inside true trim: per-side offsets
  vs render bounds are NOT absolute borders; only L-R / T-B asymmetries are
  meaningful, assuming the render crop is symmetric about print centre
  (corroborated at the 0.05mm level by the independent back measurement).
- "Equivalent ratio" for borderless faces is a convention (nominal
  both-sides margin from the game spec), not a measurement.

## QA flags

BACKGROUND_NONUNIFORM, INSUFFICIENT_MARGIN, RADIAL_DISTORTION_RISK,
CURL_SUSPECTED, PARTIAL_EDGE_SPAN, EDGE_PARTIALLY_EXCLUDED, ASPECT_DEVIATION,
WEAK_RENDER_MATCH, HIGH_REPROJECTION_ERROR, REGISTRATION_DISCREPANCY.
Sleeve caution: a perfect-fit sleeve edge can masquerade as the card edge
(+-0.3mm) - shoot unsleeved.

## Shooting guidance

Dark, matte, uniformly lit background on all four sides; telephoto lens;
card centred, filling 60-80% of frame, never touching frame edges; unsleeved;
small tilt (<3 deg) is measured and corrected; avoid glare bands crossing
edges; flatten foil curl.

## Tests

    pytest tests/unit          # geometry/statistics primitives, synthetic images
    pytest tests/regression    # Elsa 6/C2 fixture photos vs known-good values

Regression targets: back L/R 54.6% (55/45) +-1.5pts, L 2.60mm R 2.16mm
+-0.08; front x-shift 0.27mm toward left +-0.06; front-back registration
<=0.08mm. Open item: front T/B - prototype reported ~0.09mm toward top
without a back-side cross-check; this implementation measures ~0.22mm toward
bottom on the same photo (top edge sits in a documented glare band, flagged).

## Extending to other games

Add a `GameSpec` in `src/centering/games/` (card dimensions, frame-line
expectations, equivalence margins, edge-definition calibration) and a
`RenderSource` with `get_render(card_id)`. Nothing else changes.
