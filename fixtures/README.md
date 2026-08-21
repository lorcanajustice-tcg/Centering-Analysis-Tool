# Test photos (kept on this machine only)

The regression tests measure phone photos of real cards and check the
answers against known-good values. Those photos show copyrighted
Disney/Ravensburger artwork, so they are **never committed to git** — the
repository's `.gitignore` blocks this folder.

Every test that needs a photo is guarded, and skips cleanly when the photo
is not there (which is what happens on CI). To run them yourself, put the
original photos in this folder:

| file | card | used by |
|---|---|---|
| IMG_6341.HEIC / IMG_6342.HEIC | Elsa - Ice Maker 6/C2, front and back | test_elsa_front.py, test_elsa_back.py |
| IMG_6330.HEIC / IMG_6331.HEIC | Simba - Pride Protector 8/C2, front and back | test_simba_card.py |
| IMG_6344.HEIC / IMG_6345.HEIC | Elsa 6/C2, close-ups | (reference only) |
| IMG_6397.HEIC / IMG_6398.HEIC | Gadget Hackwrench 12:147, front and back, on WHITE paper | test_gadget_multibg.py |
| IMG_6399.HEIC / IMG_6400.HEIC | Gadget Hackwrench 12:147, front and back, on a dark mat with light from one side — these are the ones that should be refused | test_gadget_multibg.py |
| IMG_6401.HEIC / IMG_6402.HEIC | Gadget Hackwrench 12:147, front and back, on kraft cardboard | test_gadget_multibg.py |
| tag/T6453597{F,B}-*.jpg, tag/Y3106454{F,B}-*.jpg | Lilo & Stitch 13-244 Enchanted, two copies graded by TAG, front and back — their own report scans, cropped tight to the card | test_tag_scans.py |

The `tag/` folder holds scans that came with grading reports. Their
filenames contain the grader's own centring percentages, in the form
`<serial><F or B>-<L>L<R>R<T>T<B>B`. They are a third-party reference for
development only — the program itself never reads them. To compare our
numbers against theirs, run `calibration/tag_reference.py <folder>`.

Official card pictures are not stored here either. The program downloads
the one it needs while it runs, into `~/.cache/centering` (or wherever
`$CENTERING_CACHE` points), and the full local picture database can be
rebuilt at any time with `card_db/fetch_images.py`.
