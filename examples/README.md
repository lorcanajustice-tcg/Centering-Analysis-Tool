# examples/

`elsa_card.json` is a real combined-mode result (schema 1.0): back + front
faces, per-edge fits, uncertainties, QA flags and the front-back
registration cross-check. Numbers only - the overlay images the analyzer
also produces are git-ignored here because they contain card artwork
(see the IP note in the repository README).

Regenerated 2026-08-21 against the corrected manufactured card size
(62.9 x 87.9mm, not the 2.5 x 3.5in nominal) and the colour-edge
detectors. Absolute millimetres from before that change read about 1%
high, so do not compare this file against an older run side by side.

Interpretation reminder: front-back print registration on genuine cards
scatters by about +-0.19mm (one calibrated card reached 0.43mm), so a
nonzero `registration_mm` is not by itself evidence of a miscut.
