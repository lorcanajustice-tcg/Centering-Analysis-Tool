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
