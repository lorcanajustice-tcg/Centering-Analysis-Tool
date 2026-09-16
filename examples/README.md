# examples/

`elsa_card.json` is a real result file, produced by checking both sides of
one card. It shows exactly what the program saves: the numbers for each
face, how each edge was fitted, the margins of error, the warnings, and the
front-versus-back comparison.

Numbers only. The marked-up pictures the program also produces are kept out
of the repository, because they show card artwork — see the licensing note
in the main README.

**When it was made:** regenerated 2026-09-16 for v0.5.0, which added the
front's `hex_check` block (the ink-cost hexagon cross-check) and
`result.method`; every other number is the same as the v0.4.0 file of
2026-09-15. v0.4.0 traces the back's gold frame line on a denser grid,
which moved this card's back L/R by a tenth of a point, and the file also
picks up the plain-English wording from v0.3.1. The card size correction (real manufactured
62.9 × 87.9mm, 2026-08-21) is in too: don't compare this file side by side
with one made before that date.

**One thing to know when reading it:** `registration_mm` is how much the
front and back disagree about where the card was cut. On genuine cards that
disagreement is normally around 0.19mm, and one card measured here reached
0.43mm — the two faces are printed in separate passes. A non-zero figure
there is not on its own evidence of a miscut.
