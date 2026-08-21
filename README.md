# Card Centering Checker

Take a photo of a trading card. This tells you how well centred it is —
in millimetres, and in the usual grading style like `55/45`.

It is accurate to about a tenth of a millimetre, which is close enough to
tell a 55/45 card from a 53/47 one. It was built for Disney Lorcana and
can be extended to other card games.

---

## Getting started (Windows)

Double-click **`run_analyzer.bat`**.

The first time you run it, it spends a minute or two installing what it
needs. You need Python 3 on the computer first — get it from
[python.org](https://www.python.org/downloads/) and tick **"Add python.exe
to PATH"** when the installer offers.

After that a page opens in your browser. Drop in your photos, press
**Check this card**, and read the answer.

Everything happens on your own computer. The only thing that leaves it is
a request for the official picture of the card you are checking.

---

## Taking a photo that works

Most bad results come from the photo, not the card. Five things matter:

1. **Take the card out of the sleeve.** A snug sleeve edge looks just like
   a card edge and can throw the answer off by 0.3mm.
2. **Put it on plain white paper.** That gives the clearest edge of
   anything. A plain dark mat also works. Busy or patterned surfaces make
   it harder.
3. **Soft light from every side.** This is the big one. A single lamp, or
   a window off to one side, casts a shadow along the card's edge, and
   that shadow can move the answer by up to 0.7mm. Bounce the light off a
   wall or ceiling, or shoot in even daylight away from a window.
4. **Straight down, card flat.** A slight tilt is fine — that gets
   corrected automatically. Flatten any curl first; foil cards curl most.
5. **Leave a gap around the card.** Fill most of the frame, but keep a
   clear margin — roughly a finger's width — between the card and the edge
   of the photo. Phone lenses bend the picture near the edges, and that is
   not corrected for.

Scans work too, and so do the cropped card images that grading companies
include with a report. You do not have to do anything different for those
— it works out for itself that the card fills the frame. The only thing
it cannot handle is a card cropped so tightly there is no margin at all,
because then there is no edge to find. A millimetre of surround is plenty.

---

## Reading the answer

**`55/45`** is how graders write centring. It means one border takes 55%
of the space and the other takes 45%. 50/50 is perfect. The first number
is always the left border, or the top one.

**"Give or take 0.03mm"** is how far out the answer could be. If it says
0.52mm give or take 0.03mm, the real figure is almost certainly somewhere
between 0.49mm and 0.55mm.

**≈ (estimated)** means that edge was too unclear to measure properly, so
it was worked out a rougher way instead. It is honest but approximate —
treat it as a ballpark, and take a better photo if the number matters.

**"Not measured"** means the photo did not show enough to give an honest
answer, and it tells you why and what to try. Nothing is ever quietly
guessed.

**The marked-up picture** at the bottom of the results shows exactly where
it decided your card's edges and printed frame were. Glance at it. If a
line does not sit on the real edge, that side's number is wrong, and the
fix is a better photo.

### The two sides measure different things

The **back** of a Lorcana card has a printed gold frame inside a plain
border, so it measures the four border widths directly.

The **front** of an Enchanted card has artwork running right to the edge,
so there are no borders to measure. Instead it downloads the official
picture of that card, lines your photo up against it, and reports how far
the printing sits off centre — then converts that into the same `55/45`
style so you can compare.

If you check both sides at once, it also compares them. They are looking
at the same single cut from opposite sides, so they should agree. A small
disagreement is normal, though: the two faces are printed in separate
passes, and real cards typically differ by about 0.19mm — one card
measured here reached 0.43mm. A difference on that scale is ordinary
manufacturing, not a miscut.

---

## When it cannot find the card

Sometimes it can't work out where the card is — a dark card on a dark
surface, a busy background, a photo of somebody else's card that you can't
retake.

When that happens the page offers **"Show me where the card is"**. Drag a
rough box along the card's edges and press go. The box only tells it where
to look; the actual measuring is unchanged, and results come out the same
to within seven thousandths of a millimetre. It is there for photos you
cannot retake, not as a normal step.

---

## Using it from the command line

Installing:

    pip install -e .          # needs Python 3.10 or newer

Running:

    centering back  photo.heic                                # back of a card
    centering front photo.heic --card 8-210                   # front of a card
    centering card --back b.heic --front f.heic --card 8-210  # both, compared

Useful extras: `--out <folder>` to choose where the marked-up pictures go,
`--json <file>` to save every number.

**Naming a card.** `8-210` is set 8, card 210. Enchanted and promo cards
use their own group: `C2-6`, `P1-42`. You can also just type part of the
name, like `"Elsa - Ice Maker"`. The older `6/C2` and `7:69` styles still
work.

Using it from your own Python code:

    from centering import analyze_back, analyze_borderless, analyze_card
    r = analyze_back("back.heic", LORCANA)
    r.ratio_lr.display        # "55/45"
    r.borders_mm["left"]      # value, margin of error, and how sure it is

---

## How careful it is about being wrong

Every number comes with a margin of error, and every number says how well
it is actually known. There are three levels and they are never blurred
together:

- **Measured** — read straight off the photo.
- **Estimated** — the strict method refused this edge, but a looser,
  clearly-labelled method could recover it. Always shown with a ≈ and
  always carrying a bigger margin of error. If that margin grows past
  about the width of a grading band, the answer is refused instead.
- **Not measured** — with the reason, in plain words, and what to try.

It also watches for things that quietly ruin measurements and tells you
about them: uneven lighting, a shadow along an edge, a bent card, the card
sitting too close to the edge of the photo, a poor match against the
official picture, and the card coming out the wrong shape (usually a
sleeve, a bend, or an edge found in the wrong place).

One thing no warning can catch: a sleeve that fits perfectly can pass
itself off as the card edge, worth about 0.3mm. Always shoot unsleeved.

Two more details worth knowing:

- **Tilt is corrected.** Lens distortion is not, which is why the card
  needs a margin around it.
- **Coloured backgrounds are measured by colour, not brightness.** A card
  casts a shadow onto its background, and that shadow is a change in
  brightness that spills outside the real edge. Colour is not fooled by
  it. This is what makes grading-company scans — which sit on a coloured
  backing — measurable at all.

---

## Card pictures and licensing

The MIT licence covers the code in this repository only.

Card artwork, names and official pictures belong to Disney and
Ravensburger, and none of it is distributed here. The program downloads
the one picture it needs while it runs; the full local picture database is
rebuilt on your own machine with `card_db/fetch_images.py`; and the test
photos stay on the machine they were taken on (see `fixtures/README.md`).

---

## For developers

    pytest tests/unit          # fast, no photos needed
    pytest tests/regression    # real photos vs known-good answers

All the user-facing wording lives in one file, `src/centering/plain.py`:
the heading and the "what to do" line for every warning, and the plain
phrases for every reason a reading gets thrown out. Change wording there,
not scattered through the pipeline.

To add another card game, write a `GameSpec` in `src/centering/games/`
(card size, what the printed frame looks like, margins, edge calibration)
and a `RenderSource` with a `get_render(card_id)`. Nothing else changes.

`DEV-NOTES.md` has the detail on how the measurement actually works, and
`TODO.md` has what is still open.
