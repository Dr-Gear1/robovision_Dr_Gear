"""
robovision/utils/convolution.py
=================================
True 2-D convolution and spatial filtering for grayscale and RGB images.

Public API
----------
validate_kernel    — kernel shape, type, and content checks
convolve2d         — true 2-D convolution (grayscale, with kernel flip)
filter2d           — cross-correlation / filtering (no kernel flip)
spatial_filter     — 2-D filtering for grayscale AND RGB images (3.5)

All boundary handling is delegated to :func:`robovision.utils.padding.pad_image`,
keeping the padding strategy as a single configurable parameter.

Only NumPy — no OpenCV, no scipy.
"""

from __future__ import annotations
import numpy as np
from robovision.utils.padding import pad_image


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
            f"'{name}' must be 2-D (H, W) or 3-D (H, W, C), "
            f"got shape {image.shape}."
        )
    if image.size == 0:
        raise ValueError(f"'{name}' must not be empty.")


# ══════════════════════════════════════════════════════════════════════════════
# Kernel validation
# ══════════════════════════════════════════════════════════════════════════════

def validate_kernel(kernel: np.ndarray) -> None:
    """
    Validate a convolution kernel.

    Checks performed:
    1. Must be a numpy.ndarray.
    2. Must be 2-D.
    3. Must be non-empty.
    4. Both dimensions must be **odd** (required for symmetric padding).
    5. Must have a **numeric** dtype (integer or floating point).

    Parameters
    ----------
    kernel : np.ndarray
        Kernel to validate.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If kernel is not ndarray, or has a non-numeric dtype.
    ValueError
        If kernel is not 2-D, is empty, or has even-sized dimensions.

    Examples
    --------
    >>> validate_kernel(np.ones((3, 3)))          # passes silently
    >>> validate_kernel(np.ones((4, 4)))          # raises ValueError
    >>> validate_kernel([[1,0],[0,1]])             # raises TypeError
    """
    if not isinstance(kernel, np.ndarray):
        raise TypeError(
            f"'kernel' must be numpy.ndarray, got {type(kernel).__name__}. "
            "Convert your list to np.array first."
        )
    if kernel.ndim != 2:
        raise ValueError(
            f"'kernel' must be 2-D, got shape {kernel.shape}."
        )
    if kernel.size == 0:
        raise ValueError("'kernel' must not be empty.")
    if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise ValueError(
            f"'kernel' dimensions must both be odd integers, "
            f"got shape {kernel.shape}. "
            "Odd sizes ensure a well-defined centre pixel for symmetric padding."
        )
    if not np.issubdtype(kernel.dtype, np.number):
        raise ValueError(
            f"'kernel' must have a numeric dtype (int or float), "
            f"got {kernel.dtype}."
        )


# ══════════════════════════════════════════════════════════════════════════════
# 3.4  True 2-D Convolution (grayscale)
# ══════════════════════════════════════════════════════════════════════════════

def convolve2d(
    image: np.ndarray,
    kernel: np.ndarray,
    padding_mode: str = "reflect",
    constant_value: float = 0.0,
) -> np.ndarray:
    """
    Perform true 2-D discrete convolution on a grayscale image.

    True convolution flips the kernel 180° (both axes) before sliding it
    over the image.  For symmetric kernels (Gaussian, mean, Laplacian)
    the result is identical to cross-correlation.  For asymmetric kernels
    (Sobel, Prewitt) the flip matters.

    Algorithm
    ---------
    1. **Validate** kernel (odd, 2-D, numeric, non-empty).
    2. **Pad** the image by pad = (k//2) on each side using *padding_mode*.
    3. **Flip** the kernel 180°: ``kernel[::-1, ::-1]``.
    4. Use ``np.lib.stride_tricks.as_strided`` to build a **4-D view**
       of all (H × W) overlapping (kH × kW) windows simultaneously.
       No Python loop over pixels.
    5. Multiply element-wise with the flipped kernel and sum over the
       two window axes → output shape (H, W).

    Boundary handling
    -----------------
    All border padding is delegated to :func:`robovision.utils.padding.pad_image`.
    Available modes: 'zero', 'reflect', 'replicate', 'constant', 'circular'.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image, shape (H, W), any numeric dtype.
        Use :func:`spatial_filter` for RGB images.
    kernel : np.ndarray
        2-D convolution kernel.  Must be 2-D, both dimensions odd,
        non-empty, and numeric.  Need not be normalised.
    padding_mode : str, optional
        Border handling.  Default 'reflect'.
        Passed to :func:`robovision.utils.padding.pad_image`.
    constant_value : float, optional
        Fill value when padding_mode='constant'.  Default 0.0.

    Returns
    -------
    np.ndarray
        Convolution result, shape (H, W), dtype float32.
        Same spatial size as input ('same' output).

    Raises
    ------
    TypeError
        If image or kernel are not ndarray, or kernel has non-numeric dtype.
    ValueError
        If image is not 2-D, or kernel fails validation.

    Notes
    -----
    * Memory: the stride-tricks view is O(H·W·kH·kW).  For very large
      kernels (k > 25) on large images, prefer separable 1-D convolution.
    * For cross-correlation (no flip), use :func:`filter2d`.

    Examples
    --------
    >>> k   = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32)  # Sobel X
    >>> Gx  = convolve2d(gray, k)
    >>> box = np.ones((5,5), np.float32) / 25
    >>> blurred = convolve2d(gray, box, padding_mode='zero')
    """
    _validate_image(image)
    validate_kernel(kernel)
    if image.ndim != 2:
        raise ValueError(
            f"'image' must be 2-D for convolve2d, got shape {image.shape}. "
            "For RGB images use spatial_filter()."
        )

    img    = image.astype(np.float32)
    kH, kW = kernel.shape
    pH, pW = kH // 2, kW // 2

    # Step 3 — flip kernel 180° (true convolution vs cross-correlation)
    k_flip = kernel[::-1, ::-1].astype(np.float32)

    # Step 2 — pad with the largest needed pad on each axis
    padded = pad_image(img, max(pH, pW), mode=padding_mode,
                       constant_value=constant_value)

    # Trim padded to exact per-axis padding (handles non-square kernels)
    p = max(pH, pW)
    padded = padded[p - pH : p - pH + img.shape[0] + 2 * pH,
                    p - pW : p - pW + img.shape[1] + 2 * pW]

    H, W = img.shape

    # Step 4 — stride-trick view: shape (H, W, kH, kW), zero copies
    out_shape = (H, W, kH, kW)
    strides   = (
        padded.strides[0], padded.strides[1],
        padded.strides[0], padded.strides[1],
    )
    windows = np.lib.stride_tricks.as_strided(
        padded, shape=out_shape, strides=strides
    )

    # Step 5 — element-wise multiply and sum: fully vectorised
    return (windows * k_flip).sum(axis=(2, 3)).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 3.4 variant — Cross-correlation / filter (no flip)
# ══════════════════════════════════════════════════════════════════════════════

def filter2d(
    image: np.ndarray,
    kernel: np.ndarray,
    padding_mode: str = "reflect",
    constant_value: float = 0.0,
) -> np.ndarray:
    """
    Apply a 2-D filter via cross-correlation (no kernel flip).

    Identical to :func:`convolve2d` except the kernel is **not** flipped.
    For symmetric kernels (Gaussian, mean) the results are the same.
    For asymmetric kernels, cross-correlation and convolution differ.

    Use cross-correlation when:
    - Matching a template to an image.
    - The kernel is already specified in the 'flipped' convention.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image, shape (H, W).
    kernel : np.ndarray
        2-D filter kernel (applied without flipping).
    padding_mode : str
        Border handling.  Default 'reflect'.
    constant_value : float
        Fill value for mode='constant'.  Default 0.0.

    Returns
    -------
    np.ndarray
        Filtered image, shape (H, W), dtype float32.

    Examples
    --------
    >>> out = filter2d(gray, template_kernel)
    """
    _validate_image(image)
    validate_kernel(kernel)
    if image.ndim != 2:
        raise ValueError(
            f"'image' must be 2-D for filter2d, got shape {image.shape}."
        )

    img    = image.astype(np.float32)
    kH, kW = kernel.shape
    pH, pW = kH // 2, kW // 2
    k      = kernel.astype(np.float32)  # no flip

    padded = pad_image(img, max(pH, pW), mode=padding_mode,
                       constant_value=constant_value)
    p = max(pH, pW)
    padded = padded[p - pH : p - pH + img.shape[0] + 2 * pH,
                    p - pW : p - pW + img.shape[1] + 2 * pW]

    H, W  = img.shape
    out_shape = (H, W, kH, kW)
    strides   = (
        padded.strides[0], padded.strides[1],
        padded.strides[0], padded.strides[1],
    )
    windows = np.lib.stride_tricks.as_strided(
        padded, shape=out_shape, strides=strides
    )
    return (windows * k).sum(axis=(2, 3)).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 3.5  2-D Spatial Filtering — grayscale + RGB dispatcher
# ══════════════════════════════════════════════════════════════════════════════

def spatial_filter(
    image: np.ndarray,
    kernel: np.ndarray,
    padding_mode: str = "reflect",
    constant_value: float = 0.0,
    rgb_strategy: str = "per_channel",
) -> np.ndarray:
    """
    Apply convolution-based 2-D filtering to grayscale or RGB images.

    This is the **main entry point** for spatial filtering in RoboVision.
    It handles both grayscale and colour images with a documented strategy.

    RGB strategies
    --------------
    +----------------+-------------------------------------------------------+
    | 'per_channel'  | Apply the kernel independently to each channel.      |
    | (default)      | R, G, B are filtered separately and recombined.      |
    |                | Correct for linear filters (mean, Gaussian, Sobel).  |
    |                | Simple, fast, and the most common approach.          |
    +----------------+-------------------------------------------------------+
    | 'luminance'    | Convert to grayscale, filter, return 2-D result.     |
    |                | Use when only intensity edges are needed (e.g. Sobel  |
    |                | on a colour image for edge detection).               |
    +----------------+-------------------------------------------------------+

    Parameters
    ----------
    image : np.ndarray
        Input image, shape (H, W) for grayscale or (H, W, C) for colour.
        Any numeric dtype.  Values are converted to float32 internally.
    kernel : np.ndarray
        2-D convolution kernel.  Validated by :func:`validate_kernel`:
        must be 2-D, both dims odd, non-empty, numeric.
    padding_mode : str, optional
        Border handling strategy.  Default 'reflect'.
        Options: 'zero', 'reflect', 'replicate', 'constant', 'circular'.
    constant_value : float, optional
        Fill value when padding_mode='constant'.  Default 0.0.
    rgb_strategy : {'per_channel', 'luminance'}, optional
        How to handle colour images.  Default 'per_channel'.
        Ignored for grayscale inputs.

    Returns
    -------
    np.ndarray
        Filtered image, dtype float32.
        - 'per_channel': same shape (H, W, C) as input.
        - 'luminance'  : shape (H, W) — grayscale result.
        - Grayscale in: shape (H, W) always.

    Raises
    ------
    TypeError
        If image or kernel are not ndarray, or kernel dtype is non-numeric.
    ValueError
        If image shape is invalid, kernel fails validation,
        or rgb_strategy is not recognised.

    Notes
    -----
    * :func:`convolve2d` (true convolution with kernel flip) is used
      internally.  For symmetric kernels this is identical to correlation.
    * The kernel is NOT normalised internally — pass a normalised kernel
      (e.g. divided by its sum) if you want energy-preserving filtering.

    Examples
    --------
    >>> # Gaussian smoothing on RGB
    >>> gauss = np.outer(g, g) where g = exp(...)  # see gaussian_kernel
    >>> blurred_rgb  = spatial_filter(rgb_img, gauss)
    >>>
    >>> # Sobel edges on RGB — luminance strategy
    >>> sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], np.float32)
    >>> edges = spatial_filter(rgb_img, sobel_x, rgb_strategy='luminance')
    """
    _STRATEGIES = ("per_channel", "luminance")
    if rgb_strategy not in _STRATEGIES:
        raise ValueError(
            f"'rgb_strategy' must be one of {_STRATEGIES}, "
            f"got '{rgb_strategy}'."
        )
    _validate_image(image)
    validate_kernel(kernel)

    img = image.astype(np.float32)

    # ── Grayscale path ────────────────────────────────────────────────
    if img.ndim == 2:
        return convolve2d(img, kernel, padding_mode, constant_value)

    # ── Colour path ───────────────────────────────────────────────────
    if rgb_strategy == "luminance":
        # Convert to grayscale first (BT.601 weights)
        w    = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)
        gray = (img[:, :, :3] * w).sum(axis=2)
        return convolve2d(gray, kernel, padding_mode, constant_value)

    else:  # per_channel — apply independently to every channel
        channels = [
            convolve2d(img[:, :, c], kernel, padding_mode, constant_value)
            for c in range(img.shape[2])
        ]
        return np.stack(channels, axis=2)
