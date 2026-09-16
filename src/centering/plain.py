"""Every word this program says to a person lives here.

The rest of the code works in technical terms - scan lines, residuals,
homographies. None of that belongs on a screen someone is trying to read.
So each warning carries a short code (a stable name used by the saved
JSON file and by the tests), and this file turns that code into:

  title   - a short, plain heading:  "Uneven lighting"
  advice  - what the person should actually do about it

The sentence in the middle, the one with the numbers in it, is written at
the place the problem is detected. The rule there is the same: short
words, real numbers, no jargon.
"""
from __future__ import annotations

# --- warnings -------------------------------------------------------------
# code -> (plain heading, what to do about it)

FLAGS: dict[str, tuple[str, str]] = {
    "BACKGROUND_NONUNIFORM": (
        "Uneven lighting",
        "Light the card evenly from all sides. Move the lamp further away, "
        "or bounce it off a wall or ceiling, and take the photo again."),
    "CURL_SUSPECTED": (
        "The card looks bent",
        "Flatten the card and re-shoot. Foil cards curl the most, and a "
        "curled card makes one border look wider than it really is."),
    "ASPECT_DEVIATION": (
        "The card is not the shape it should be",
        "Take the card out of its sleeve, flatten it, and shoot it square "
        "on. If the shape is still wrong, one of the four edges is being "
        "found in the wrong place - check the overlay picture."),
    "FRAME_LINE_SPARSE": (
        "The printed gold line was hard to follow",
        "Nothing to do - the extra doubt is already in the plus-or-minus "
        "figure. A sharper photo usually makes the line easier to follow."),
    "PARTIAL_EDGE_SPAN": (
        "Only part of this edge could be measured",
        "Use a plain background with a bit more contrast against the card, "
        "and light it evenly, so the whole edge stands out."),
    "MANUAL_LOCALIZATION": (
        "You marked the card position by hand",
        "Nothing to do. Your box only says where to look - the measuring "
        "is done exactly the same way as normal."),
    "EDGE_ESTIMATED": (
        "One edge is an estimate, not a clean measurement",
        "Treat this number as approximate. A cleaner photo - plain white "
        "paper, even light, no sleeve - usually turns it into a proper "
        "measurement."),
    "EDGE_RELOCATED": (
        "One edge was found on a second look",
        "Nothing to do - it was measured normally. If you drew the box "
        "yourself, a tighter box around the card saves the second look."),
    "EDGE_INFERRED": (
        "One edge was worked out from the other three",
        "The other direction is unaffected. For a proper reading in this "
        "direction too, re-shoot so that edge stands out from the "
        "background - plain white paper, even light, no sleeve."),
    "EDGE_PARTIALLY_EXCLUDED": (
        "Part of this edge had to be skipped",
        "Glare, a shadow, or a background too close in colour to the card. "
        "Re-shoot with softer light and a plainer background."),
    "WEAK_RENDER_MATCH": (
        "Poor match against the official card picture",
        "Check the card ID is the right card. If it is, re-shoot in focus, "
        "square on, with even light."),
    "HIGH_REPROJECTION_ERROR": (
        "The official card picture did not line up cleanly",
        "Usually a blurred, tilted or glary photo. Re-shoot flat, in "
        "focus, and square on."),
    "RENDER_SPAN_MISMATCH": (
        "One edge of the card was found in the wrong place",
        "Almost always a shadow, a glare band or a curled corner being "
        "mistaken for the edge. Re-shoot on plain white paper with soft "
        "light from all sides."),
    "RENDER_CROP_BIAS_CORRECTED": (
        "A known offset in the official picture was corrected",
        "Nothing to do. This correction is automatic and already included "
        "in the result."),
    "REGISTRATION_DISCREPANCY": (
        "Front and back do not quite agree",
        "Some disagreement is normal - the two sides are printed in "
        "separate passes. Check the card is unsleeved and flat, and that "
        "both photos are sharp."),
    "COLOUR_EDGE_ADOPTED": (
        "The card edge was found by colour instead of brightness",
        "Nothing to do. On a coloured background this is the more accurate "
        "choice, because a shadow changes brightness but not colour."),
    "SHADOW_BAND_SUSPECTED": (
        "A shadow may be hiding the true edge",
        "Re-shoot with soft light coming from all sides. A single lamp, or "
        "a window on one side, casts a shadow along the card edge that can "
        "shift the result by up to 0.7mm."),
    "TIGHT_CROP": (
        "This is a scan or a cropped picture, not a framed photo",
        "Nothing to do. It is measured the same way. Just note that if it "
        "was cropped out of a phone photo, the lens bending near the "
        "original photo edge is not corrected for."),
    "HEX_ANCHOR_USED": (
        "Measured from the ink-cost hexagon",
        "Nothing to do. Naming the card as well adds a second, independent "
        "measurement to check this one against."),
    "HEX_ANCHOR_DISAGREES": (
        "The two front measurements disagree",
        "Look at the overlay picture: the top or left edge is probably in "
        "the wrong place. Re-shoot on plain white paper with soft light "
        "from all sides."),
    "HEX_SCALE_MISMATCH": (
        "The ink-cost hexagon is the wrong size for this card",
        "Take the card out of its sleeve and re-shoot flat. If it still "
        "happens, check the overlay picture for an edge in the wrong "
        "place."),
    "HEX_LAYOUT_ASSUMED": (
        "The hexagon's printed position was assumed",
        "If the card is from one of the sets named, give its set and "
        "number so the right position is used."),
    "HEX_LAYOUT_CHOSEN": (
        "Two hexagon designs fitted; one was picked by its look",
        "If the result looks wrong, give the card's set and number so the "
        "right design is used."),
    "HEX_CHECK_SKIPPED": (
        "The ink-cost hexagon cross-check did not run",
        "Nothing to do - the main result does not depend on it."),
    "RADIAL_DISTORTION_RISK": (
        "The card is too close to the edge of the photo",
        "Move back, or re-frame, so there is a clear gap - about a "
        "finger's width - between every edge of the card and the edge of "
        "the photo. Phone lenses bend the picture near the edges."),
}

DEFAULT_ADVICE = (
    "Re-shoot the card flat and unsleeved on plain white paper, with soft "
    "light from all sides, and try again.")


def title_for(code: str) -> str:
    """Short plain heading for a warning code."""
    if code in FLAGS:
        return FLAGS[code][0]
    return code.replace("_", " ").capitalize()


def advice_for(code: str) -> str:
    """What the person should do about this warning."""
    if code in FLAGS:
        return FLAGS[code][1]
    return DEFAULT_ADVICE


# --- why a single scan line was thrown away -------------------------------
# These come from the edge detectors. They are counted up and shown as
# "3 lines: a glare band was in the way", so the phrase has to read as the
# tail of that sentence.

REJECTS: dict[str, str] = {
    "band_truncated": "the search ran off the side of the photo",
    "window_outside_frame": "the search ran off the side of the photo",
    "insufficient_texture_contrast":
        "the background looked too much like the card",
    "insufficient_contrast":
        "too little brightness difference between card and background",
    "no_sustained_crossing": "no clear edge there",
    "no_crossing": "no clear edge there",
    "no_sustained_departure": "no clear edge there",
    "glare_band_skipped": "a glare band was in the way",
    "shadowed_outside_level": "a shadow beside the card got in the way",
    "non_monotonic_at_edge": "the edge did not change cleanly enough to read",
    "shallow_departure_slope": "the edge faded in too gradually to pin down",
    "insufficient_chroma": "the background had too little colour to use",
    "chroma_step_below_noise": "the colour change at the edge was too faint",
    "no_card_plateau": "the card's own colour was not steady enough to read",
    "no_peak_above_threshold": "the printed line was too faint to find",
    "min_peak_rescued": "the printed line was very faint",
    "degenerate_peak": "the printed line could not be pinned down",
    "off_modal_frame_distance":
        "the printed line turned up at an odd distance and was ignored",
}


def reject_phrase(token: str) -> str:
    """Plain phrase for one detector reject reason."""
    return REJECTS.get(token, token.replace("_", " "))
