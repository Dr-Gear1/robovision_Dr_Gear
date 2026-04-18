"""
robovision/filters/edge_detection.py
======================================
Edge detection and image analysis techniques.

Public API
----------
sobel_gradients   — Gx, Gy, magnitude, angle via Sobel kernels
bit_plane_slice   — extract a single bit-plane from a uint8 image
bit_plane_all     — return all 8 bit-planes stacked
canny             — full Canny edge detector pipeline

Only NumPy — no OpenCV, no scipy.
"""

from __future__ import annotations
import numpy as np
from robovision.filters.filters import (
    _validate_image, _to_float32, _to_gray,
    convolve2d, apply_filter, gaussian_filter,
)


# ══════════════════════════════════════════════════════════════════════════════
# 4.5  Sobel gradients
# ══════════════════════════════════════════════════════════════════════════════

# Standard 3×3 Sobel kernels
_SOBEL_X = np.array([[-1,  0,  1],
                     [-2,  0,  2],
                     [-1,  0,  1]], dtype=np.float32)

_SOBEL_Y = np.array([[-1, -2, -1],
                     [ 0,  0,  0],
                     [ 1,  2,  1]], dtype=np.float32)


def sobel_gradients(
    image: np.ndarray,
    padding_mode: str = "reflect",
) -> dict[str, np.ndarray]:
    """
    Compute Sobel gradient images.

    Applies the 3×3 Sobel operator to estimate the horizontal (Gx) and
    vertical (Gy) image gradients.  From these, gradient magnitude and
    orientation are derived.

    Sobel kernels
    -------------
    Kx = [[-1, 0, 1],    Ky = [[-1,-2,-1],
          [-2, 0, 2],          [ 0, 0, 0],
          [-1, 0, 1]]          [ 1, 2, 1]]

    Derived quantities:

        magnitude(r,c) = sqrt( Gx² + Gy² )
        angle(r,c)     = arctan2(Gy, Gx)   in degrees, range [-180, 180]

    Parameters
    ----------
    image : np.ndarray
        Input image, shape (H, W) or (H, W, C).  3-D inputs are converted
        to grayscale automatically before gradient computation.
        dtype float [0, 1] or uint8 [0, 255].
    padding_mode : str
        Border handling for convolution.  Default 'reflect'.

    Returns
    -------
    dict with keys:
        'Gx'        : np.ndarray (H, W) float32 — horizontal gradient
        'Gy'        : np.ndarray (H, W) float32 — vertical gradient
        'magnitude' : np.ndarray (H, W) float32 — gradient magnitude
        'angle'     : np.ndarray (H, W) float32 — gradient angle in degrees

    Raises
    ------
    TypeError
        If image is not ndarray.
    ValueError
        If image has invalid shape.

    Notes
    -----
    * Magnitude is NOT normalised — values may exceed [0, 1].
      Normalise with ``mag / mag.max()`` for display.
    * Angle is in degrees, unsigned for edge direction use: ``angle % 180``.
    * For pre-smoothing before Sobel, apply :func:`gaussian_filter` first.

    Examples
    --------
    >>> result = sobel_gradients(gray_img)
    >>> mag    = result['magnitude']
    >>> angle  = result['angle']
    """
    _validate_image(image)
    img = _to_float32(_to_gray(image))

    Gx  = convolve2d(img, _SOBEL_X, padding_mode)
    Gy  = convolve2d(img, _SOBEL_Y, padding_mode)
    mag = np.sqrt(Gx ** 2 + Gy ** 2).astype(np.float32)
    ang = np.rad2deg(np.arctan2(Gy, Gx)).astype(np.float32)

    return {"Gx": Gx, "Gy": Gy, "magnitude": mag, "angle": ang}


# ══════════════════════════════════════════════════════════════════════════════
# 4.6  Bit-plane slicing
# ══════════════════════════════════════════════════════════════════════════════

def bit_plane_slice(
    image: np.ndarray,
    plane: int,
) -> np.ndarray:
    """
    Extract a single bit-plane from a grayscale image.

    Each pixel value (0–255) is represented as 8 binary bits.  Bit-plane k
    contains the k-th bit of every pixel.  Plane 7 (MSB) captures the
    dominant structure; plane 0 (LSB) is nearly random noise.

    Math
    ----
    For each pixel value v:

        bit_plane[k](r,c) = (v >> k) & 1

    Parameters
    ----------
    image : np.ndarray
        Grayscale image, shape (H, W).  Accepts float [0, 1] or uint8.
        Float inputs are scaled to [0, 255] and cast to uint8 internally.
        3-D inputs are converted to grayscale automatically.
    plane : int
        Bit-plane index in [0, 7].  0 = LSB (least significant),
        7 = MSB (most significant, dominant structure).

    Returns
    -------
    np.ndarray
        Binary bit-plane image, shape (H, W), dtype uint8,
        values in {0, 1}.

    Raises
    ------
    TypeError
        If image is not ndarray.
    ValueError
        If image shape is invalid or plane not in [0, 7].

    Notes
    -----
    * Bit-planes 4–7 contain most of the visual information.
    * Planes 0–3 are dominated by noise and quantisation effects.
    * Useful for watermarking, compression analysis, and data hiding.

    Examples
    --------
    >>> msb = bit_plane_slice(gray, plane=7)   # most significant
    >>> lsb = bit_plane_slice(gray, plane=0)   # least significant
    """
    _validate_image(image)
    if not isinstance(plane, int):
        raise TypeError(f"'plane' must be int, got {type(plane).__name__}.")
    if not (0 <= plane <= 7):
        raise ValueError(f"'plane' must be in [0, 7], got {plane}.")

    img = _to_gray(image)

    # Convert float → uint8 if needed
    if img.dtype != np.uint8:
        img_f = img.astype(np.float32)
        if img_f.max() <= 1.0:
            img_f = img_f * 255.0
        img = np.clip(img_f, 0, 255).astype(np.uint8)

    # Bitwise shift and mask — fully vectorised
    return ((img >> plane) & 1).astype(np.uint8)


def bit_plane_all(image: np.ndarray) -> np.ndarray:
    """
    Extract all 8 bit-planes from a grayscale image.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image, shape (H, W), float [0, 1] or uint8.

    Returns
    -------
    np.ndarray
        Stacked bit-planes, shape (8, H, W), dtype uint8.
        Index 0 = LSB (plane 0), index 7 = MSB (plane 7).

    Examples
    --------
    >>> planes = bit_plane_all(gray)
    >>> planes.shape
    (8, 480, 640)
    >>> msb = planes[7]
    """
    _validate_image(image)
    return np.stack([bit_plane_slice(image, p) for p in range(8)], axis=0)


# ══════════════════════════════════════════════════════════════════════════════
# 4.8  Canny edge detector
# ══════════════════════════════════════════════════════════════════════════════

def _non_maximum_suppression(magnitude: np.ndarray, angle: np.ndarray) -> np.ndarray:
    """
    Thin edges to one pixel width by suppressing non-maximum gradient pixels.

    For each pixel, the gradient direction determines which two neighbours
    to compare against.  If the pixel's magnitude is not the local maximum
    along the gradient direction, it is suppressed to 0.

    Loop justification
    ------------------
    NMS requires comparing each pixel against its two directional neighbours.
    The neighbour coordinates depend on the quantised angle at that pixel,
    making a pure vectorised form non-trivial.  The implementation uses
    vectorised boolean masks for each of the four quantised directions
    (0°, 45°, 90°, 135°), avoiding explicit per-pixel Python loops.
    """
    H, W   = magnitude.shape
    result = np.zeros_like(magnitude)
    angle_deg = angle % 180.0   # unsigned direction

    # Pad magnitude to handle borders without if-statements
    padded = np.pad(magnitude, 1, mode='constant', constant_values=0)

    # Quantise angles into 4 directions and use vectorised masks
    # Direction 0°  → compare left/right (East/West)
    # Direction 45° → compare NE/SW
    # Direction 90° → compare up/down (North/South)
    # Direction 135°→ compare NW/SE

    for direction, (dr1, dc1, dr2, dc2) in enumerate([
        (0, -1, 0,  1),   # 0°
        (-1, 1, 1, -1),   # 45°
        (-1, 0, 1,  0),   # 90°
        (-1,-1, 1,  1),   # 135°
    ]):
        if direction == 0:
            mask = (angle_deg < 22.5) | (angle_deg >= 157.5)
        elif direction == 1:
            mask = (angle_deg >= 22.5) & (angle_deg < 67.5)
        elif direction == 2:
            mask = (angle_deg >= 67.5) & (angle_deg < 112.5)
        else:
            mask = (angle_deg >= 112.5) & (angle_deg < 157.5)

        # Neighbour values via padded slicing — vectorised
        r_idx = np.where(mask)
        if r_idx[0].size == 0:
            continue
        rows, cols = r_idx
        pr, pc = rows + 1, cols + 1    # padded coords

        n1 = padded[pr + dr1, pc + dc1]
        n2 = padded[pr + dr2, pc + dc2]
        m  = magnitude[rows, cols]

        keep = (m >= n1) & (m >= n2)
        result[rows[keep], cols[keep]] = m[keep]

    return result


def _hysteresis(nms: np.ndarray, low: float, high: float) -> np.ndarray:
    """
    Apply double-threshold hysteresis to connect edge fragments.

    Strong pixels (> high) are definite edges.
    Weak pixels (low < val ≤ high) are edges only if connected to strong.
    Below-low pixels are suppressed.

    Uses iterative dilation — justified because connectivity tracking
    inherently requires knowing which pixels have been visited.  The loop
    runs at most min(H, W) times; in practice 2–5 iterations suffice.
    """
    strong = nms >= high
    weak   = (nms >= low) & (nms < high)
    edges  = strong.copy()

    # Iterative dilation of strong edges into connected weak pixels
    # Each iteration expands the strong set by one pixel in all 8 directions
    prev_count = -1
    while True:
        count = edges.sum()
        if count == prev_count:
            break
        prev_count = count
        # 8-neighbour expansion — fully vectorised via slicing
        dilated = (
            edges[:-2, :-2] | edges[:-2, 1:-1] | edges[:-2, 2:] |
            edges[1:-1,:-2] |                     edges[1:-1, 2:] |
            edges[2:, :-2]  | edges[2:, 1:-1]  | edges[2:, 2:]
        )
        edges[1:-1, 1:-1] |= (dilated & weak[1:-1, 1:-1])

    return edges.astype(np.float32)


def canny(
    image: np.ndarray,
    low_thresh: float = 0.05,
    high_thresh: float = 0.15,
    gaussian_size: int = 5,
    gaussian_sigma: float = 1.0,
    padding_mode: str = "reflect",
) -> np.ndarray:
    """
    Detect edges using the full Canny edge detection pipeline.

    The Canny detector is a multi-stage algorithm designed to find a
    wide range of edges while being robust to noise.

    Pipeline stages
    ---------------
    1. **Gaussian smoothing** — reduce noise before gradient computation.
    2. **Sobel gradients** — compute Gx, Gy, magnitude, angle.
    3. **Non-maximum suppression (NMS)** — thin edges to 1 pixel wide
       by suppressing pixels that are not the local maximum in the
       gradient direction.
    4. **Double thresholding** — classify pixels as strong (> high),
       weak (low ≤ val ≤ high), or suppressed (< low).
    5. **Hysteresis edge tracking** — promote weak pixels to edges if
       they are 8-connected to at least one strong pixel.

    Parameters
    ----------
    image : np.ndarray
        Input image, shape (H, W) or (H, W, C).  3-D inputs converted
        to grayscale.  dtype float [0, 1] or uint8 [0, 255].
    low_thresh : float
        Lower hysteresis threshold in [0, 1].  Default 0.05.
        Pixels below this are always suppressed.
    high_thresh : float
        Upper hysteresis threshold in [0, 1].  Default 0.15.
        Pixels above this are always kept.  Must be > low_thresh.
    gaussian_size : int
        Size of the Gaussian smoothing kernel.  Default 5 (odd).
    gaussian_sigma : float
        Sigma of the Gaussian smoothing.  Default 1.0.
    padding_mode : str
        Padding mode for convolution.  Default 'reflect'.

    Returns
    -------
    np.ndarray
        Binary edge map, shape (H, W), dtype float32,
        values in {0.0 (no edge), 1.0 (edge)}.

    Raises
    ------
    TypeError
        If image is not ndarray.
    ValueError
        If low_thresh >= high_thresh, or either is outside [0, 1].

    Notes
    -----
    * Good starting values: low=0.05, high=0.15 for most natural images.
    * Increase gaussian_sigma if the result is too noisy.
    * The ratio high/low ≈ 2–3 is Canny's original recommendation.

    Examples
    --------
    >>> edges = canny(img, low_thresh=0.05, high_thresh=0.15)
    """
    _validate_image(image)
    if not (0.0 <= low_thresh <= 1.0 and 0.0 <= high_thresh <= 1.0):
        raise ValueError("low_thresh and high_thresh must both be in [0, 1].")
    if low_thresh >= high_thresh:
        raise ValueError(
            f"'low_thresh' ({low_thresh}) must be < 'high_thresh' ({high_thresh})."
        )

    # Stage 1 — Gaussian smoothing
    img = _to_float32(_to_gray(image))
    smoothed = gaussian_filter(img, size=gaussian_size,
                               sigma=gaussian_sigma,
                               padding_mode=padding_mode)

    # Stage 2 — Sobel gradients
    grads = sobel_gradients(smoothed, padding_mode)
    mag   = grads["magnitude"]
    ang   = grads["angle"]

    # Normalise magnitude to [0, 1] for threshold comparison
    mag_max = mag.max()
    if mag_max > 0:
        mag_norm = mag / mag_max
    else:
        return np.zeros_like(img)

    # Stage 3 — Non-maximum suppression
    nms = _non_maximum_suppression(mag_norm, ang)

    # Stages 4 & 5 — Double threshold + hysteresis
    edges = _hysteresis(nms, low_thresh, high_thresh)

    return edges
