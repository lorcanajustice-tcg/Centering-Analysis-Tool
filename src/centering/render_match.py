"""SIFT feature matching of a photo to the official card render."""
from __future__ import annotations

import cv2
import numpy as np


# SIFT on a very large photo costs memory quadratically in the long edge
# (a 27MP scan needs several GB and can be killed outright), and buys
# nothing: the match is limited by the render's own resolution (~1500px
# across a 63mm card), not the photo's. Photos above this are matched
# downscaled and the homography is rescaled back to full-resolution photo
# coordinates - the edge scans that carry the measurement still run at
# full resolution. The cap sits above a 12MP phone photo, so ordinary
# captures are matched exactly as before.
MATCH_MAX_LONG_EDGE = 4096


def match_to_render(photo_gray: np.ndarray, render_gray: np.ndarray,
                    photo_mask: np.ndarray | None = None,
                    n_features: int = 8000, ratio: float = 0.72,
                    ransac_px: float = 3.0,
                    max_long_edge: int = MATCH_MAX_LONG_EDGE):
    """Returns (H_photo_to_render, n_inliers, median_reproj_px).

    Deterministic: OpenCV RNG is seeded before RANSAC.
    """
    k = 1.0
    long_edge = max(photo_gray.shape[:2])
    if max_long_edge and long_edge > max_long_edge:
        k = max_long_edge / float(long_edge)
        photo_gray = cv2.resize(photo_gray, None, fx=k, fy=k,
                                interpolation=cv2.INTER_AREA)
        if photo_mask is not None:
            photo_mask = cv2.resize(photo_mask, (photo_gray.shape[1],
                                                 photo_gray.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
    p8 = cv2.convertScaleAbs(photo_gray)
    r8 = cv2.convertScaleAbs(render_gray)
    sift = cv2.SIFT_create(nfeatures=n_features)
    kp1, des1 = sift.detectAndCompute(p8, photo_mask)
    kp2, des2 = sift.detectAndCompute(r8, None)
    if des1 is None or des2 is None or len(kp1) < 50 or len(kp2) < 50:
        raise RuntimeError(
            "There is not enough detail in this photo to match it against "
            "the official picture of the card. Re-shoot it sharply in "
            "focus, filling most of the frame, with even light.")
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in knn if m.distance < ratio * n.distance]
    if len(good) < 30:
        raise RuntimeError(
            f"Only {len(good)} points matched between your photo and the "
            "official picture of the card. Either this is not that card - "
            "check the card ID - or the photo is too blurred, or too washed "
            "out by glare, to line up.")
    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    cv2.setRNGSeed(20260703)
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, ransac_px)
    if H is None or inl is None or inl.sum() < 25:
        raise RuntimeError(
            "Your photo and the official picture of the card could not be "
            "lined up. Check the card ID is right, then re-shoot sharply "
            "in focus, square on, with even light.")
    inl = inl.ravel().astype(bool)
    proj = cv2.perspectiveTransform(src[inl], H).reshape(-1, 2)
    err = np.linalg.norm(proj - dst[inl].reshape(-1, 2), axis=1)
    if k != 1.0:
        # H maps downscaled-photo -> render; compose with the downscale so
        # the caller keeps a full-resolution-photo -> render homography
        H = H @ np.diag([k, k, 1.0])
    return H, int(inl.sum()), float(np.median(err))
