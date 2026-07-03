# Test fixture photos (local only)

The regression tests use phone photos of real cards. Card faces reproduce
Disney/Ravensburger copyrighted artwork, so the photos are **not tracked in
git** and must never be committed (see the repo `.gitignore`).

All photo-based regression tests are guarded with
`pytest.mark.skipif(not FIXTURE.exists(), ...)` and skip cleanly when the
photos are absent (e.g. on CI). To run them, place the original photos in
this folder:

| file | card | used by |
|---|---|---|
| IMG_6341.HEIC / IMG_6342.HEIC | Elsa - Ice Maker 6/C2 front/back | test_elsa_front.py, test_elsa_back.py |
| IMG_6330.HEIC / IMG_6331.HEIC | Simba - Pride Protector 8/C2 front/back | test_simba_card.py |
| IMG_6344.HEIC / IMG_6345.HEIC | Elsa 6/C2 close-ups | (reference) |

Official card renders are never stored in the repo either: the analyzer
downloads the render it needs at runtime into `~/.cache/centering`
(or `$CENTERING_CACHE`), and the full local render database can be rebuilt
any time with `card_db/fetch_images.py`.
