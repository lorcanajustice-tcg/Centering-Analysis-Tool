"""SIFT feature matching of a photo to the official card render."""
from __future__ import annotations

import cv2
import numpy as np


def match_to_render(photo_gray: np.ndarray, render_gray: np.ndarray,
                    photo_mask: np.ndarray | None = None,
                    n_features: int = 8000, ratio: float = 0.72,
                    ransac_px: float = 3.0):
    """Returns (H_photo_to_render, n_inliers, median_reproj_px).

    Deterministic: OpenCV RNG is seeded before RANSAC.
    """
    p8 = cv2.convertScaleAbs(photo_gray)
    r8 = cv2.convertScaleAbs(render_gray)
    sift = cv2.SIFT_create(nfeatures=n_features)
    kp1, des1 = sift.detectAndCompute(p8, photo_mask)
    kp2, des2 = sift.detectAndCompute(r8, None)
    if des1 is None or des2 is None or len(kp1) < 50 or len(kp2) < 50:
        raise RuntimeError("insufficient SIFT features for render matching")
    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(des1, des2, k=2)
    good = [m for m, n in knn if m.distance < ratio * n.distance]
    if len(good) < 30:
        raise RuntimeError(
            f"only {len(good)} ratio-test matches; photo may not show this card "
            "(check card id) or is too blurred/glared for feature matching")
    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    cv2.setRNGSeed(20260703)
    H, inl = cv2.findHomography(src, dst, cv2.RANSAC, ransac_px)
    if H is None or inl is None or inl.sum() < 25:
        raise RuntimeError("RANSAC homography failed or too few inliers")
    inl = inl.ravel().astype(bool)
    proj = cv2.perspectiveTransform(src[inl], H).reshape(-1, 2)
    err = np.linalg.norm(proj - dst[inl].reshape(-1, 2), axis=1)
    return H, int(inl.sum()), float(np.median(err))
