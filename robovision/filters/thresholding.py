"""
robovision/filters/thresholding.py
====================================
Image thresholding techniques — three methods covering the full spectrum
from simple to fully automatic.

Public API
----------
threshold_global    — fixed-value binary threshold
threshold_otsu      — automatic threshold via Otsu's method
threshold_adaptive  — locally computed threshold (mean or Gaussian)

Only NumPy — no OpenCV, no scipy.
"""

from __future__ import annotations
import numpy as np
from robovision.filters.filters import (
    _validate_image, _validate_kernel_size, _to_float32, _to_gray,
    pad_image, gaussian_kernel,
)


# ══════════════════════════════════════════════════════════════════════════════
# 4.4a  Global thresholding
# ══════════════════════════════════════════════════════════════════════════════

def threshold_global(
    image: np.ndarray,
    thresh: float,
    mode: str = "binary",
) -> np.ndarray:
    """
    Apply a fixed global threshold to a grayscale image.

    All pixels above *thresh* are set to the 'high' value and all pixels
    at or below *thresh* are set to the 'low' value (or vice-versa for
    inverse modes).

    Threshold modes
    ---------------
    +------------------+----------------------------------------+
    | Mode             | Rule                                   |
    +==================+========================================+
    | 'binary'         | pixel > thresh → 1.0, else → 0.0      |
    +------------------+----------------------------------------+
    | 'binary_inv'     | pixel > thresh → 0.0, else → 1.0      |
    +------------------+----------------------------------------+
    | 'trunc'          | pixel > thresh → thresh, else → pixel  |
    +------------------+----------------------------------------+
    | 'tozero'         | pixel > thresh → pixel, else → 0.0    |
    +------------------+----------------------------------------+
    | 'tozero_inv'     | pixel > thresh → 0.0, else → pixel    |
    +------------------+----------------------------------------+

    Parameters
    ----------
    image : np.ndarray
        Grayscale image, shape (H, W), float [0, 1] or uint8 [0, 255].
        3-D inputs are converted to grayscale automatically.
    thresh : float
        Threshold value in [0, 1] for float images, or [0, 255] for uint8.
        Values outside the image's range produce a blank or full output.
    mode : str
        One of the five threshold modes above.  Default 'binary'.

    Returns
    -------
    np.ndarray
        Binary (or modified) image, shape (H, W), dtype float32.

    Raises
    ------
    TypeError
        If image is not ndarray.
    ValueError
        If image has wrong shape, thresh is not in [0, 1], or mode invalid.

    Notes
    -----
    * thresh should be in [0, 1] — uint8 images are auto-normalised before
      comparison.
    * Fully vectorised — no Python loops.

    Examples
    --------
    >>> binary = threshold_global(gray, thresh=0.5)
    >>> inv    = threshold_global(gray, thresh=0.5, mode='binary_inv')
    """
    _MODES = ("binary", "binary_inv", "trunc", "tozero", "tozero_inv")
    if mode not in _MODES:
        raise ValueError(f"'mode' must be one of {_MODES}, got '{mode}'.")
    _validate_image(image)
    if not isinstance(thresh, (int, float)):
        raise TypeError(f"'thresh' must be numeric, got {type(thresh).__name__}.")
    if not (0.0 <= thresh <= 1.0):
        raise ValueError(f"'thresh' must be in [0, 1], got {thresh}.")

    img = _to_float32(_to_gray(image))

    if mode == "binary":
        return (img > thresh).astype(np.float32)
    elif mode == "binary_inv":
        return (img <= thresh).astype(np.float32)
    elif mode == "trunc":
        return np.where(img > thresh, thresh, img).astype(np.float32)
    elif mode == "tozero":
        return np.where(img > thresh, img, 0.0).astype(np.float32)
    else:  # tozero_inv
        return np.where(img <= thresh, img, 0.0).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 4.4b  Otsu's automatic threshold
# ══════════════════════════════════════════════════════════════════════════════

def threshold_otsu(
    image: np.ndarray,
    n_bins: int = 256,
    return_thresh: bool = False,
) -> np.ndarray | tuple[np.ndarray, float]:
    """
    Automatically compute and apply Otsu's threshold.

    Otsu's method finds the threshold that **maximises the between-class
    variance** of the two pixel classes (foreground / background).  It is
    equivalent to minimising the weighted sum of within-class variances.

    Math
    ----
    For each candidate threshold t:

        ω₀(t) = P(pixel ≤ t)             (background weight)
        ω₁(t) = 1 − ω₀(t)               (foreground weight)
        μ₀(t), μ₁(t)                     (class means)
        σ²_B(t) = ω₀·ω₁·(μ₀ − μ₁)²     (between-class variance)

    Optimal t* = argmax σ²_B(t)

    The computation is vectorised over all candidate thresholds at once
    using cumulative sums — no Python loop over thresholds.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image, shape (H, W) or (H, W, C), any dtype.
        3-D inputs are converted to grayscale automatically.
    n_bins : int, optional
        Number of histogram bins used to discretise pixel values.
        Default 256.  Fewer bins → faster but coarser threshold.
    return_thresh : bool, optional
        If True, also return the computed threshold value.
        Default False.

    Returns
    -------
    binary : np.ndarray
        Binary image, shape (H, W), dtype float32, values in {0.0, 1.0}.
    thresh : float
        (Only if return_thresh=True) Optimal threshold in [0, 1].

    Raises
    ------
    TypeError
        If image is not ndarray.
    ValueError
        If image has wrong shape or n_bins < 2.

    Notes
    -----
    * Works best on bimodal histograms (clear foreground/background).
    * For multimodal histograms consider multi-level Otsu or adaptive
      thresholding.

    Examples
    --------
    >>> binary = threshold_otsu(gray_img)
    >>> binary, t = threshold_otsu(gray_img, return_thresh=True)
    >>> print(f'Otsu threshold: {t:.3f}')
    """
    _validate_image(image)
    if not isinstance(n_bins, int) or n_bins < 2:
        raise ValueError(f"'n_bins' must be an int >= 2, got {n_bins}.")

    img = _to_float32(_to_gray(image))

    # Build normalised histogram
    hist, bin_edges = np.histogram(img.ravel(), bins=n_bins, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    prob = hist / hist.sum()                        # normalised probabilities
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Cumulative sums — vectorised over all thresholds simultaneously
    w0 = np.cumsum(prob)                            # background weight
    w1 = 1.0 - w0                                  # foreground weight
    mu0 = np.cumsum(prob * bin_centers) / (w0 + 1e-10)
    mu_total = (prob * bin_centers).sum()
    mu1 = (mu_total - np.cumsum(prob * bin_centers)) / (w1 + 1e-10)

    sigma_b_sq = w0 * w1 * (mu0 - mu1) ** 2       # between-class variance

    # Optimal threshold
    opt_idx = np.argmax(sigma_b_sq)
    thresh  = float(bin_centers[opt_idx])

    binary = (img > thresh).astype(np.float32)
    if return_thresh:
        return binary, thresh
    return binary


# ══════════════════════════════════════════════════════════════════════════════
# 4.4c  Adaptive thresholding
# ══════════════════════════════════════════════════════════════════════════════

def threshold_adaptive(
    image: np.ndarray,
    block_size: int = 11,
    C: float = 0.02,
    method: str = "mean",
    sigma: float = 2.0,
) -> np.ndarray:
    """
    Apply adaptive (locally computed) thresholding.

    Instead of a single global threshold, each pixel is compared against
    a *local* threshold computed from its neighbourhood.  This handles
    uneven illumination far better than global methods.

    Adaptive threshold computation
    ------------------------------
    For each pixel (r, c):

        local_thresh = local_stat(neighbourhood) − C

    where *local_stat* is:
        - **'mean'**     : arithmetic mean of the block (fast)
        - **'gaussian'** : Gaussian-weighted mean (smoother, less noise)

    Then:  output[r,c] = 1.0  if  image[r,c] > local_thresh  else 0.0

    Parameters
    ----------
    image : np.ndarray
        Grayscale image, shape (H, W) or (H, W, C), any dtype.
        3-D inputs converted to grayscale automatically.
    block_size : int
        Size of the local neighbourhood (must be odd).  Default 11.
        Larger → smoother threshold map but less local sensitivity.
    C : float
        Constant subtracted from the local mean/Gaussian average.
        Fine-tunes the sensitivity.  Default 0.02.
        Positive C → threshold is slightly below the local mean
        (fewer pixels pass).
    method : {'mean', 'gaussian'}
        How to compute the local statistic.  Default 'mean'.
    sigma : float
        Gaussian sigma used when method='gaussian'.  Default 2.0.
        Ignored for method='mean'.

    Returns
    -------
    np.ndarray
        Binary image, shape (H, W), dtype float32, values in {0.0, 1.0}.

    Raises
    ------
    TypeError
        If image is not ndarray.
    ValueError
        If image shape is invalid, block_size is not a positive odd int,
        or method is not 'mean' or 'gaussian'.

    Notes
    -----
    * Implemented via convolution — fully vectorised, no per-pixel loops.
    * For document binarisation: method='gaussian', block_size=15, C=0.02.
    * For natural images: method='mean', block_size=21, C=0.05.

    Examples
    --------
    >>> binary = threshold_adaptive(gray, block_size=11, C=0.02)
    >>> binary = threshold_adaptive(gray, method='gaussian', sigma=2.0)
    """
    _METHODS = ("mean", "gaussian")
    if method not in _METHODS:
        raise ValueError(f"'method' must be one of {_METHODS}, got '{method}'.")
    _validate_image(image)
    _validate_kernel_size(block_size, "block_size")
    if not isinstance(C, (int, float)):
        raise TypeError(f"'C' must be numeric, got {type(C).__name__}.")

    img = _to_float32(_to_gray(image))
    H, W = img.shape
    pad = block_size // 2

    if method == "mean":
        # Uniform kernel — box filter for local mean
        k = np.ones((block_size, block_size), dtype=np.float32) / (block_size ** 2)
    else:
        # Gaussian-weighted local mean
        k = gaussian_kernel(block_size, sigma)

    # Convolve to get local mean at every pixel — fully vectorised
    padded = pad_image(img, pad, mode="reflect")
    shape   = (H, W, block_size, block_size)
    strides = (
        padded.strides[0], padded.strides[1],
        padded.strides[0], padded.strides[1],
    )
    windows    = np.lib.stride_tricks.as_strided(padded, shape=shape,
                                                  strides=strides)
    local_mean = (windows * k).sum(axis=(2, 3))     # (H, W)

    local_thresh = local_mean - C
    return (img > local_thresh).astype(np.float32)
