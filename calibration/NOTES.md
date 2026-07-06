# Render-crop bias calibration notes (2026-07-03)

## The problem

Full-art (borderless) front T/B centering was wildly off: Simba - Pride
Protector 8/C2 read equiv T/B 62/38 while its back read ~50/50, implying a
front-back print registration of -0.72mm. Implausible: x-registration on
the same card measured 0.009mm.

## Root cause

The borderless pipeline assumed the official render is cropped
symmetrically about the print centre. That assumption was only ever
cross-validated on the x axis (Elsa 6/C2). It is FALSE vertically:
Ravensburger renders (1468x2048) are cropped ~1.15mm total in width
(symmetric) but ~2.3mm total in height with more taken off the TOP.

## Evidence

1. Anchor survey over all 3,211 renders (card_db/anchors_report.json):
   the crop is layout-locked - footer emblem y=1899-1900 (+-1px) on
   bordered AND full-art cards; standard-frame border 60px L/R (symmetric),
   frame top 66px, footer band top 1907px. Exceptions with their own
   (internally consistent) layouts: Location cards, Q1/Q2 promos, 16
   foil-border cards - listed in the report.
2. Four standard-frame front+back pairs (12:60, 12:91, 10:189, 12:191,
   photos 2026-07-03): vertical bias = raw_render_shift - back_derived_shift
   = +0.045 / +0.111 / +0.187 / +0.379 mm -> +0.18 +- 0.06 (sem),
   card-to-card scatter +-0.13 (includes true front-back registration).
   Method: top cut edge + known card height + group render scale
   (23.545 px/mm from the two cleanest L/R pairs); backs measured
   frame-relative. See calibration/measure_crop_bias.py and *_{front,back}.json.
3. The Simba pair itself reads higher (+0.4..0.7) but BOTH its photos have
   a shadow band along one horizontal edge that displaces the pipeline
   step/texture scanners outward by ~0.7mm (verified visually: the
   pipeline top edge sat in the shadow/fiber zone ~0.7mm above the true
   cut). Its back was therefore never really 50/50 either. Treated as a
   compromised measurement, not as evidence for a second crop constant.

## What changed in the code

- GameSpec gains render_crop_bias_mm / render_crop_bias_unc_mm.
- LORCANA: bias y = +0.20mm (unc 0.20, kept honest until more clean pairs
  exist), x = 0.0 (unc 0.05). equiv_margin_tb_mm placeholder 6.0 replaced
  with measured ~4.6.
- borderless.py subtracts the bias, carries the systematic in the
  uncertainty, and emits QA flag RENDER_CROP_BIAS_CORRECTED.
- New regression fixture: Simba 8/C2 pair (tests/regression/test_simba_card.py).

## Known residual issues / next steps

1. Shadow-band edge artifact (~0.5-0.7mm, worst on dark textured mats with
   directional light): step_scan can lock onto the outer shadow boundary,
   texture_scan onto the shadow-smoothed zone. Mitigations to implement:
   specular-ridge / plateau-knee hybrid detector (prototype in
   calibration/measure_crop_bias.py cut_scan()), and an aspect-ratio QA
   gate on the borderless quad like back.py has.
2. Capture protocol matters more than code: photograph on a bright matte
   background (white paper), diffuse light, card small-ish in frame
   (>=5% clearance to photo edges). All four sides then become clean
   bright/dark steps and the shadow artifact vanishes.
3. To tighten the y-bias below +-0.1: 3-4 more standard front+back pairs
   photographed per (2), plus at least one clean Enchanted pair to close
   the question whether full-art renders share exactly the same constant
   (all pixel-anchor evidence says they do).
4. card_db/ holds all 3,211 renders + index.json; fetch_images.py
   refreshes it after new set releases (re-run the anchor survey then).

---

# Recalibration under the improved protocol (2026-07-06)

## Data

Five front+back pairs on WHITE paper, diffuse light (IMG_6397/98 Gadget
12:147; IMG_6403/04 Julieta 12:24; IMG_6405/06 Dangerous Plan 12:133;
IMG_6407/08 Sabotage 12:96; IMG_6409/10 Woody 12-54-P3 full-art/promo).
Measured with the library pipelines themselves (polarity-agnostic
detectors, zero-bias GameSpec for the fronts) - NOT the v1/v2 prototype
script. Per-pair numbers: reshoot_2026_07_06.json.

## Flip convention (established, was an unstated assumption)

Back photos are related to front photos by a VERTICAL-AXIS flip: L/R
mirrors, T/B does not (card_report.py's assumption - now confirmed).
Evidence: the two strong-y pairs (Dangerous Plan, Sabotage: front y -0.49
/ -0.32) match their backs' frame-derived y (-0.31 / -0.34) in the SAME
direction (registration 0.09 / 0.02mm); the mirrored hypothesis would
imply ~0.8mm registrations. All five back photos share one orientation
(SIFT rotation check, +-2 deg).

## Result

bias_y = mean(front_raw_y - back_derived_y) = -0.08 +- 0.07 (sem),
card scatter +-0.16. CONSISTENT WITH ZERO. Standard-frame-only subset:
-0.04 +- 0.07. Full-art (Woody): -0.26, within ~1.1 sigma of the mean ->
no evidence for a separate full-art constant (as the anchor survey
predicts). Adopted: y = -0.08, unc 0.10.

bias_x pair-estimator read -0.13 +- 0.09, but a true x bias is excluded
by the anchor survey (60/60px symmetric frame in all standard renders);
attributed to front-back print registration noise, which is larger than
assumed in 2026-07-03: per-card scatter +-0.19mm, one pair (Julieta)
0.43mm. x kept at 0.0 +- 0.05.

Equivalence margins from the five backs: L+R total 4.775 +- 0.017 (sd) -
manufacture-constant to 20 microns - and T+B 4.305 +- 0.078. GameSpec
updated to 4.78 / 4.31 (old 4.76 / 4.6; the 4.6 T/B figure was
shadow-inflated).

## Why this supersedes the 2026-07-03 value (+0.18 +- 0.06)

The old shoot was dark-desk with detectors since shown vulnerable to the
shadow-band artifact, whose mechanism (directional light displacing
horizontal-edge scans outward, worst at the top) produces exactly a
POSITIVE-only y-bias contamination - the same artifact that excluded the
Simba pair from that calibration. The new protocol eliminates it (clean
step edges on white; hybrid cross-check quiet on all ten photos).

Consequence for Simba (8/C2): with the honest constant its front now
reads y +0.52 / equiv T/B 62/38 and front-back y registration -0.53 -
i.e. the pair's documented ~0.5mm artifact is VISIBLE instead of being
half-masked by a bias constant that had absorbed the same artifact.
test_simba_card.py now regression-tests exactly that behaviour.

## Caveat (documented, accepted)

The pair estimator cannot separate the render-crop asymmetry from any
off-centre of the printed back frame (delta_frame): the constant is
operationally bias_render - delta_frame. That is the RIGHT constant for
the analyzer's purpose (front/back consistency), but it should not be
quoted as a pure render-pipeline property. Mean back frame-vs-cut offsets
over the five cards: x -0.03mm, y -0.15mm (top border < bottom border on
average) - either the frame sits high in the back design or these cards'
die cuts sit high; indistinguishable with n=5 from one print run.
