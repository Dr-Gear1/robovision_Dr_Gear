"""
robovision/filters/histogram_ops.py
=====================================
Histogram-based image processing operations.

Public API
----------
compute_histogram        — pixel intensity histogram
histogram_equalization   — enhance contrast via CDF stretching
histogram_matching       — match histogram of one image to another
gamma_correction         — power-law intensity transformation

Only NumPy — no OpenCV, no scipy.
"""

from __future__ import annotations
import numpy as np
from robovision.filters.filters import _validate_image, _to_float32, _to_gray


# ══════════════════════════════════════════════════════════════════════════════
# 4.7a  Histogram computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_histogram(
    image: np.ndarray,
    n_bins: int = 256,
    normalize: bool = False,
    channel: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the intensity histogram of an image.

    Parameters
    ----------
    image : np.ndarray
        Input image, shape (H, W) for grayscale or (H, W, C) for color.
        dtype float [0, 1] or uint8 [0, 255].
    n_bins : int, optional
        Number of histogram bins.  Default 256.
    normalize : bool, optional
        If True, return a probability density (sums to 1.0).
        Default False (raw pixel counts).
    channel : int or None, optional
        For 3-D images: which channel to compute (0=R, 1=G, 2=B).
        If None (default), converts to grayscale first.

    Returns
    -------
    hist : np.ndarray
        Histogram counts (or densities if normalize=True).
        Shape (n_bins,), dtype float32.
    bin_centers : np.ndarray
        Centre value of each bin, shape (n_bins,), dtype float32.
        Range [0, 1] for float images.

    Raises
    ------
    TypeError
        If image is not ndarray.
    ValueError
        If image has invalid shape, n_bins < 1, or channel is out of range.

    Notes
    -----
    * For a full colour histogram use a loop over channels or
      :func:`extract_color_histogram` in the features module.
    * Fully vectorised — uses np.histogram.

    Examples
    --------
    >>> hist, bins = compute_histogram(gray_img, n_bins=256)
    >>> hist, bins = compute_histogram(gray_img, normalize=True)
    """
    _validate_image(image)
    if not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError(f"'n_bins' must be int >= 1, got {n_bins}.")

    img = _to_float32(image)

    if img.ndim == 3:
        if channel is not None:
            if not (0 <= channel < img.shape[2]):
                raise ValueError(
                    f"'channel' {channel} out of range for image with "
                    f"{img.shape[2]} channels."
                )
            pixels = img[:, :, channel].ravel()
        else:
            pixels = _to_gray(img).ravel()
    else:
        pixels = img.ravel()

    counts, edges = np.histogram(pixels, bins=n_bins, range=(0.0, 1.0))
    counts      = counts.astype(np.float32)
    bin_centers = ((edges[:-1] + edges[1:]) / 2).astype(np.float32)

    if normalize:
        total = counts.sum()
        if total > 0:
            counts /= total

    return counts, bin_centers


# ══════════════════════════════════════════════════════════════════════════════
# 4.7b  Histogram equalization
# ══════════════════════════════════════════════════════════════════════════════

def histogram_equalization(
    image: np.ndarray,
    n_bins: int = 256,
) -> np.ndarray:
    """
    Enhance image contrast by equalising the intensity histogram.

    Histogram equalization redistributes pixel intensities so that the
    output histogram is approximately uniform, maximising the use of the
    available intensity range.

    Math
    ----
    Let p(k) = normalised histogram (probability of intensity k).

    The Cumulative Distribution Function (CDF):

        CDF(k) = Σ_{i=0}^{k} p(i)

    Mapping function:

        output(r,c) = CDF( input(r,c) )    ∈ [0, 1]

    This stretches the CDF to be linear, flattening the histogram.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image, shape (H, W), float [0, 1] or uint8 [0, 255].
        3-D inputs are converted to grayscale automatically.
    n_bins : int, optional
        Histogram resolution.  Default 256.  Higher = finer mapping.

    Returns
    -------
    np.ndarray
        Contrast-enhanced image, shape (H, W), dtype float32, range [0, 1].

    Raises
    ------
    TypeError
        If image is not ndarray.
    ValueError
        If image has invalid shape or n_bins < 2.

    Notes
    -----
    * Works on grayscale only.  For colour images apply channel-by-channel
      or convert to HSV and equalise only the V channel.
    * May over-enhance images that already have good contrast.

    Examples
    --------
    >>> enhanced = histogram_equalization(gray_img)
    """
    _validate_image(image)
    if not isinstance(n_bins, int) or n_bins < 2:
        raise ValueError(f"'n_bins' must be int >= 2, got {n_bins}.")

    img = _to_float32(_to_gray(image))

    # Build histogram and CDF — vectorised
    hist, edges = np.histogram(img.ravel(), bins=n_bins, range=(0.0, 1.0))
    prob  = hist.astype(np.float64) / hist.sum()
    cdf   = np.cumsum(prob)                     # CDF: shape (n_bins,)

    # Map each pixel through the CDF using np.interp — fully vectorised
    bin_centers = (edges[:-1] + edges[1:]) / 2
    equalized   = np.interp(img.ravel(), bin_centers, cdf)

    return equalized.reshape(img.shape).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 4.9  Histogram matching
# ══════════════════════════════════════════════════════════════════════════════

def histogram_matching(
    source: np.ndarray,
    reference: np.ndarray,
    n_bins: int = 256,
) -> np.ndarray:
    """
    Match the histogram of *source* to that of *reference*.

    Also called histogram specification.  Finds a monotone mapping
    function T such that the output image has approximately the same
    intensity distribution as the reference.

    Math
    ----
    Let CDF_s and CDF_r be the CDFs of source and reference respectively.

    For each intensity level u in source:

        T(u) = CDF_r⁻¹( CDF_s(u) )

    where CDF_r⁻¹ is the (approximate) inverse CDF of the reference,
    implemented here by nearest-neighbour lookup in CDF_r.

    Parameters
    ----------
    source : np.ndarray
        Image to be transformed, shape (H, W) or (H, W, C).
        float [0, 1] or uint8.
    reference : np.ndarray
        Target distribution image, shape (H', W') or (H', W', C).
        Does not need to be the same size as source.
    n_bins : int, optional
        Number of bins for CDF computation.  Default 256.

    Returns
    -------
    np.ndarray
        Transformed image with matched histogram.
        Same shape as *source*, dtype float32, range [0, 1].

    Raises
    ------
    TypeError
        If source or reference are not ndarray.
    ValueError
        If shapes are invalid or n_bins < 2.

    Notes
    -----
    * For colour images, matching is applied independently to each channel
      (RGB matching) or to the V channel in HSV space for more natural results.
    * A perfect match is only possible if both images have the same number
      of distinct intensity levels.

    Examples
    --------
    >>> matched = histogram_matching(source_img, reference_img)
    """
    _validate_image(source, "source")
    _validate_image(reference, "reference")
    if not isinstance(n_bins, int) or n_bins < 2:
        raise ValueError(f"'n_bins' must be int >= 2, got {n_bins}.")

    def _match_channel(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """Match a single channel — returns float32 [0,1]."""
        src_f = _to_float32(src)
        ref_f = _to_float32(ref)

        # Source CDF
        hist_s, edges_s = np.histogram(src_f.ravel(), bins=n_bins, range=(0.0, 1.0))
        cdf_s = np.cumsum(hist_s.astype(np.float64))
        cdf_s /= cdf_s[-1]                          # normalise to [0, 1]
        bin_c_s = (edges_s[:-1] + edges_s[1:]) / 2

        # Reference CDF
        hist_r, edges_r = np.histogram(ref_f.ravel(), bins=n_bins, range=(0.0, 1.0))
        cdf_r = np.cumsum(hist_r.astype(np.float64))
        cdf_r /= cdf_r[-1]
        bin_c_r = (edges_r[:-1] + edges_r[1:]) / 2

        # Compute T(u) = CDF_r⁻¹( CDF_s(u) ) for each bin centre in source
        # Use np.interp with the reference CDF as the lookup table
        # For each value u: find cdf_s(u), then find the bin in ref where cdf_r ≈ cdf_s(u)
        cdf_s_at_src = np.interp(src_f.ravel(), bin_c_s, cdf_s)
        mapped       = np.interp(cdf_s_at_src, cdf_r, bin_c_r)

        return mapped.reshape(src_f.shape).astype(np.float32)

    src_f = _to_float32(source)
    ref_f = _to_float32(reference)

    if src_f.ndim == 2:
        ref_gray = _to_gray(ref_f) if ref_f.ndim == 3 else ref_f
        return _match_channel(src_f, ref_gray)

    # 3-D: match each channel independently
    channels = []
    for c in range(src_f.shape[2]):
        ref_c = ref_f[:, :, c] if (ref_f.ndim == 3 and c < ref_f.shape[2]) \
                else _to_gray(ref_f)
        channels.append(_match_channel(src_f[:, :, c], ref_c))

    return np.stack(channels, axis=2)


# ══════════════════════════════════════════════════════════════════════════════
# 5.0  Gamma correction
# ══════════════════════════════════════════════════════════════════════════════

def gamma_correction(
    image: np.ndarray,
    gamma: float,
) -> np.ndarray:
    """
    Apply power-law (gamma) intensity transformation.

    Gamma correction adjusts the overall brightness of an image:

        output = input ^ γ         (values in [0, 1])

    - γ < 1  → **brighten** (expand dark regions, compress bright)
    - γ = 1  → **no change** (identity)
    - γ > 1  → **darken** (compress dark regions, expand bright)

    Inverse gamma (monitor correction): γ = 1 / display_gamma.
    Standard display gamma ≈ 2.2, so corrective γ ≈ 0.45.

    Parameters
    ----------
    image : np.ndarray
        Input image, shape (H, W) or (H, W, C), any numeric dtype.
        float [0, 1] or uint8 [0, 255] — normalised automatically.
    gamma : float
        Exponent γ.  Must be > 0.
        Typical values: 0.45 (screen linearisation), 2.2 (encoding),
        0.5 (strong brightening), 2.0 (strong darkening).

    Returns
    -------
    np.ndarray
        Gamma-corrected image, same shape as input, dtype float32,
        range [0, 1].

    Raises
    ------
    TypeError
        If image is not ndarray or gamma is not numeric.
    ValueError
        If gamma <= 0 or image has invalid shape.

    Notes
    -----
    * Pixels are clipped to [0, 1] before the power operation to avoid
      NaN from negative values.
    * Fully vectorised — single NumPy power call, no loops.
    * The sRGB standard uses a piecewise gamma of ~2.2 (not a pure power
      law), but the simple power law here is sufficient for most CV tasks.

    Examples
    --------
    >>> bright  = gamma_correction(img, gamma=0.5)   # brighten
    >>> dark    = gamma_correction(img, gamma=2.0)   # darken
    >>> linear  = gamma_correction(img, gamma=1.0)   # identity
    """
    _validate_image(image)
    if not isinstance(gamma, (int, float)):
        raise TypeError(f"'gamma' must be numeric, got {type(gamma).__name__}.")
    if gamma <= 0:
        raise ValueError(f"'gamma' must be > 0, got {gamma}.")

    img = _to_float32(image)
    img = np.clip(img, 0.0, 1.0)                 # guard against slight overflows

    # Single vectorised power call — no loops
    return np.power(img, gamma).astype(np.float32)
