"""
robovision/utils/pixel_clipping.py
=====================================
Pixel-value clipping and clamping operations.

Public API
----------
clip               — clip to [low, high] range
clip_percentile    — clip at custom percentile bounds
clip_sigma         — clip at N standard deviations from the mean
clip_uint8         — clamp to valid uint8 range [0, 255]

Only NumPy — no OpenCV, no scipy.
"""

from __future__ import annotations
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# Shared validation
# ══════════════════════════════════════════════════════════════════════════════

def _validate_image(image: np.ndarray, name: str = "image") -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError(
            f"'{name}' must be numpy.ndarray, got {type(image).__name__}."
        )
    if image.ndim not in (2, 3):
        raise ValueError(
            f"'{name}' must be 2-D or 3-D, got shape {image.shape}."
        )
    if image.size == 0:
        raise ValueError(f"'{name}' must not be empty.")


# ══════════════════════════════════════════════════════════════════════════════
# 3.2  Pixel clipping
# ══════════════════════════════════════════════════════════════════════════════

def clip(
    image: np.ndarray,
    low: float = 0.0,
    high: float = 1.0,
) -> np.ndarray:
    """
    Clip pixel values to the range [low, high].

    Any pixel value below *low* is set to *low*; any value above *high*
    is set to *high*.  Values inside the range are unchanged.

    This is the most common clipping operation — used to:
    - Prevent overflow after arithmetic operations (addition, convolution).
    - Ensure values stay in [0, 1] after normalisation.
    - Remove outlier intensities before display.

    Formula
    -------
        output = max(low, min(high, pixel))

    Implemented as a single ``np.clip`` call — fully vectorised.

    Parameters
    ----------
    image : np.ndarray
        Input image, shape (H, W) or (H, W, C), any numeric dtype.
    low : float, optional
        Lower bound of the clipping range.  Default 0.0.
    high : float, optional
        Upper bound of the clipping range.  Default 1.0.

    Returns
    -------
    np.ndarray
        Clipped image, same shape and dtype as input.

    Raises
    ------
    TypeError
        If image is not ndarray, or low / high are not numeric.
    ValueError
        If image shape is invalid, or low >= high.

    Notes
    -----
    * The output dtype is preserved (no float conversion).
    * For display purposes, use clip(img, 0, 1) before imshow.
    * Does not modify the input array.

    Examples
    --------
    >>> safe = clip(img_after_filter, low=0.0, high=1.0)
    >>> uint8_safe = clip(img_uint8, low=0, high=255)
    """
    _validate_image(image)
    for name, val in [("low", low), ("high", high)]:
        if not isinstance(val, (int, float)):
            raise TypeError(
                f"'{name}' must be numeric, got {type(val).__name__}."
            )
    if low >= high:
        raise ValueError(
            f"'low' ({low}) must be strictly less than 'high' ({high})."
        )
    return np.clip(image, low, high)


def clip_percentile(
    image: np.ndarray,
    low_pct: float = 2.0,
    high_pct: float = 98.0,
) -> np.ndarray:
    """
    Clip pixel values at custom percentile bounds.

    Computes the *low_pct*-th and *high_pct*-th percentiles of all pixel
    values, then clips to those bounds.  This is a robust way to remove
    extreme outliers without knowing the exact data range in advance.

    Use case: satellite images, medical scans, HDR images where a small
    number of very bright or very dark pixels would dominate the range.

    Parameters
    ----------
    image : np.ndarray
        Input image, shape (H, W) or (H, W, C), any numeric dtype.
    low_pct : float, optional
        Lower percentile in [0, 100).  Default 2.0.
        Pixels below this percentile are clipped.
    high_pct : float, optional
        Upper percentile in (0, 100].  Default 98.0.
        Pixels above this percentile are clipped.

    Returns
    -------
    np.ndarray
        Clipped image, same shape, dtype float32.

    Raises
    ------
    TypeError
        If image is not ndarray.
    ValueError
        If percentiles are out of [0, 100] or low_pct >= high_pct.

    Notes
    -----
    * Percentiles are computed globally over all pixels and channels.
    * After clipping, consider :func:`normalize_minmax` to stretch
      the remaining range to [0, 1].

    Examples
    --------
    >>> clipped = clip_percentile(img, low_pct=1, high_pct=99)
    """
    _validate_image(image)
    for name, val in [("low_pct", low_pct), ("high_pct", high_pct)]:
        if not isinstance(val, (int, float)):
            raise TypeError(f"'{name}' must be numeric, got {type(val).__name__}.")
        if not (0.0 <= val <= 100.0):
            raise ValueError(
                f"'{name}' must be in [0, 100], got {val}."
            )
    if low_pct >= high_pct:
        raise ValueError(
            f"'low_pct' ({low_pct}) must be less than 'high_pct' ({high_pct})."
        )

    img = image.astype(np.float32)
    lo  = float(np.percentile(img, low_pct))
    hi  = float(np.percentile(img, high_pct))
    return np.clip(img, lo, hi).astype(np.float32)


def clip_sigma(
    image: np.ndarray,
    n_sigma: float = 3.0,
) -> np.ndarray:
    """
    Clip pixel values beyond N standard deviations from the mean.

    Computes the global mean μ and standard deviation σ, then clips to:

        [μ - n_sigma·σ,  μ + n_sigma·σ]

    This is effective for removing statistical outliers while preserving
    the bulk of the distribution.

    Parameters
    ----------
    image : np.ndarray
        Input image, shape (H, W) or (H, W, C), any numeric dtype.
    n_sigma : float, optional
        Number of standard deviations.  Default 3.0.
        Common values: 2.0 (stricter), 3.0 (standard), 5.0 (permissive).

    Returns
    -------
    np.ndarray
        Clipped image, same shape, dtype float32.

    Raises
    ------
    TypeError
        If image is not ndarray.
    ValueError
        If n_sigma <= 0.

    Examples
    --------
    >>> clipped = clip_sigma(img, n_sigma=2.5)
    """
    _validate_image(image)
    if not isinstance(n_sigma, (int, float)) or n_sigma <= 0:
        raise ValueError(f"'n_sigma' must be a positive number, got {n_sigma}.")

    img   = image.astype(np.float32)
    mu    = img.mean()
    sigma = img.std()
    lo    = mu - n_sigma * sigma
    hi    = mu + n_sigma * sigma
    return np.clip(img, lo, hi).astype(np.float32)


def clip_uint8(image: np.ndarray) -> np.ndarray:
    """
    Clamp pixel values to the valid uint8 range [0, 255] and cast.

    Convenience wrapper for the most common clipping use case —
    converting float or int images back to display-ready uint8 after
    any arithmetic operation.

    Parameters
    ----------
    image : np.ndarray
        Input image, any shape and numeric dtype.

    Returns
    -------
    np.ndarray
        Image with values clamped to [0, 255], dtype uint8.

    Examples
    --------
    >>> display = clip_uint8(filtered_img * 255)
    """
    _validate_image(image)
    return np.clip(image, 0, 255).astype(np.uint8)
