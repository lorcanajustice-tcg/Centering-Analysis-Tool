# examples/

`elsa_card.json` is a real result file, produced by checking both sides of
one card. It shows exactly what the program saves: the numbers for each
face, how each edge was fitted, the margins of error, the warnings, and the
front-versus-back comparison.

Numbers only. The marked-up pictures the program also produces are kept out
of the repository, because they show card artwork — see the licensing note
in the main README.

**When it was made:** 2026-08-21, after two changes. Card size was
corrected to the real manufactured 62.9 × 87.9mm (it had been using the
2.5 × 3.5 inch nominal size, which made every millimetre figure read about
1% high), and the colour-based edge detectors landed. Don't compare this
file side by side with one made before that date.

**One thing to know when reading it:** `registration_mm` is how much the
front and back disagree about where the card was cut. On genuine cards that
disagreement is normally around 0.19mm, and one card measured here reached
0.43mm — the two faces are printed in separate passes. A non-zero figure
there is not on its own evidence of a miscut.
