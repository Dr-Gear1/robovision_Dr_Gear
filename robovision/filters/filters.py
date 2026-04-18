"""
robovision/filters/filters.py
==============================
Core image filtering pipeline — padding, convolution, and three filters.

Public API
----------
pad_image        — pad an image with at least 3 modes
convolve2d       — true 2-D convolution (grayscale)
apply_filter     — apply any kernel to grayscale or RGB
mean_filter      — box / mean filter
gaussian_kernel  — generate a 2-D Gaussian kernel
gaussian_filter  — Gaussian smoothing filter
median_filter    — median filter (justified loops)

Only NumPy — no OpenCV, no scipy.
"""

from __future__ import annotations
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# Shared utilities (validation + dtype)
# ══════════════════════════════════════════════════════════════════════════════

def _validate_image(image: np.ndarray, name: str = "image") -> None:
    """Raise TypeError / ValueError for invalid image inputs."""
    if not isinstance(image, np.ndarray):
        raise TypeError(
            f"'{name}' must be a numpy.ndarray, got {type(image).__name__}."
        )
    if image.ndim not in (2, 3):
        raise ValueError(
            f"'{name}' must be 2-D (H, W) or 3-D (H, W, C), got shape {image.shape}."
        )


def _validate_kernel_size(k: int, name: str = "kernel_size") -> None:
    if not isinstance(k, int):
        raise TypeError(f"'{name}' must be an int, got {type(k).__name__}.")
    if k < 1 or k % 2 == 0:
        raise ValueError(
            f"'{name}' must be a positive odd integer, got {k}."
        )


def _to_float32(image: np.ndarray) -> np.ndarray:
    """Return float32 copy, normalising uint8 [0,255] → [0,1]."""
    img = image.astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0
    return img


def _to_gray(image: np.ndarray) -> np.ndarray:
    """Convert (H,W,C) → (H,W) grayscale using BT.601 weights."""
    if image.ndim == 2:
        return image
    w = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)
    return (image[:, :, :3] * w).sum(axis=2).astype(image.dtype)


# ══════════════════════════════════════════════════════════════════════════════
# 3.3  Padding  (at least 3 modes)
# ══════════════════════════════════════════════════════════════════════════════

_VALID_PAD_MODES = ("zero", "reflect", "replicate", "wrap")


def pad_image(
    image: np.ndarray,
    pad_width: int,
    mode: str = "reflect",
) -> np.ndarray:
    """
    Pad an image symmetrically on all four sides.

    Supports four padding modes:

    +-----------+--------------------------------------------------------------+
    | Mode      | Description                                                  |
    +===========+==============================================================+
    | 'zero'    | Fill border pixels with 0 (black).                           |
    |           | ``| 0 0 | a b c | 0 0 |``                                   |
    +-----------+--------------------------------------------------------------+
    | 'reflect' | Mirror the image content, excluding the edge pixel.          |
    |           | ``| c b | a b c | b c |``                                   |
    |           | Best default — avoids edge artefacts.                        |
    +-----------+--------------------------------------------------------------+
    | 'replicate'| Repeat the edge pixel.                                      |
    |           | ``| a a | a b c | c c |``                                   |
    +-----------+--------------------------------------------------------------+
    | 'wrap'    | Wrap around (periodic / circular).                           |
    |           | ``| b c | a b c | a b |``                                   |
    +-----------+--------------------------------------------------------------+

    Parameters
    ----------
    image : np.ndarray
        Input image, shape (H, W) or (H, W, C), any dtype.
    pad_width : int
        Number of pixels to add on each side.  Must be >= 0.
    mode : {'zero', 'reflect', 'replicate', 'wrap'}
        Padding strategy.  Default 'reflect'.

    Returns
    -------
    np.ndarray
        Padded image, shape (H + 2p, W + 2p) or (H + 2p, W + 2p, C).
        Same dtype as input.

    Raises
    ------
    TypeError
        If image is not ndarray or pad_width is not int.
    ValueError
        If image shape is invalid, pad_width < 0, or mode is unknown.

    Notes
    -----
    * pad_width = kernel_size // 2 gives 'same' output size after convolution.

    Examples
    --------
    >>> padded = pad_image(img, pad_width=3, mode='reflect')
    """
    _validate_image(image)
    if not isinstance(pad_width, int):
        raise TypeError(f"'pad_width' must be int, got {type(pad_width).__name__}.")
    if pad_width < 0:
        raise ValueError(f"'pad_width' must be >= 0, got {pad_width}.")
    if mode not in _VALID_PAD_MODES:
        raise ValueError(
            f"'mode' must be one of {_VALID_PAD_MODES}, got '{mode}'."
        )

    if pad_width == 0:
        return image.copy()

    np_mode_map = {
        "zero":      "constant",
        "reflect":   "reflect",
        "replicate": "edge",
        "wrap":      "wrap",
    }
    if image.ndim == 2:
        pad_spec = ((pad_width, pad_width), (pad_width, pad_width))
    else:
        pad_spec = ((pad_width, pad_width), (pad_width, pad_width), (0, 0))

    return np.pad(image, pad_spec, mode=np_mode_map[mode])


# ══════════════════════════════════════════════════════════════════════════════
# 3.4  2-D Convolution
# ══════════════════════════════════════════════════════════════════════════════

def _validate_kernel(kernel: np.ndarray) -> None:
    """Validate kernel: must be ndarray, 2-D, odd sizes, numeric, non-empty."""
    if not isinstance(kernel, np.ndarray):
        raise TypeError(
            f"'kernel' must be numpy.ndarray, got {type(kernel).__name__}."
        )
    if kernel.ndim != 2:
        raise ValueError(
            f"'kernel' must be 2-D, got shape {kernel.shape}."
        )
    if kernel.size == 0:
        raise ValueError("'kernel' must not be empty.")
    if kernel.shape[0] % 2 == 0 or kernel.shape[1] % 2 == 0:
        raise ValueError(
            f"'kernel' dimensions must both be odd, got shape {kernel.shape}."
        )
    if not np.issubdtype(kernel.dtype, np.number):
        raise ValueError(
            f"'kernel' must have a numeric dtype, got {kernel.dtype}."
        )


def convolve2d(
    image: np.ndarray,
    kernel: np.ndarray,
    padding_mode: str = "reflect",
) -> np.ndarray:
    """
    Perform true 2-D discrete convolution on a grayscale image.

    The kernel is flipped (rotated 180°) before sliding — this is true
    convolution, not cross-correlation.  For symmetric kernels (e.g.,
    Gaussian) the result is identical to cross-correlation.

    Algorithm
    ---------
    1. Pad the image by pad = (k//2) pixels using *padding_mode*.
    2. Flip the kernel 180° (both axes).
    3. Use ``np.lib.stride_tricks.as_strided`` to build a view of all
       (H × W) windows of size (kh × kw) — no Python loops.
    4. Multiply each window by the flipped kernel and sum.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image, shape (H, W), dtype float32 recommended.
    kernel : np.ndarray
        2-D convolution kernel.  Must be 2-D, odd-sized, non-empty,
        and numeric.  Need not sum to 1 (normalisation is the caller's
        responsibility).
    padding_mode : str
        Padding mode passed to :func:`pad_image`.  Default 'reflect'.

    Returns
    -------
    np.ndarray
        Convolution output, shape (H, W), dtype float32.
        Same spatial size as input ('same' convolution).

    Raises
    ------
    TypeError
        If image or kernel are not ndarray.
    ValueError
        If image is not 2-D, or kernel fails validation.

    Notes
    -----
    * For large kernels (k > ~25) separable convolution is faster; use
      :func:`gaussian_filter` which applies two 1-D passes internally.
    * The stride-tricks view shares memory with the padded array — do
      not modify it in place.

    Examples
    --------
    >>> k = np.ones((3, 3), dtype=np.float32) / 9
    >>> out = convolve2d(gray_img, k)
    """
    _validate_image(image)
    _validate_kernel(kernel)
    if image.ndim != 2:
        raise ValueError(
            f"'image' must be 2-D for convolve2d, got shape {image.shape}. "
            "For RGB, use apply_filter()."
        )

    img    = image.astype(np.float32)
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2

    # True convolution: flip kernel
    k_flipped = kernel[::-1, ::-1].astype(np.float32)

    padded = pad_image(img, max(ph, pw), mode=padding_mode)

    # Trim to exactly the needed padding per axis
    H, W = img.shape
    padded = padded[
        max(ph, pw) - ph : max(ph, pw) - ph + H + 2 * ph,
        max(pw, pw) - pw : max(pw, pw) - pw + W + 2 * pw,
    ]
    # Rebuild with exact per-axis padding
    padded = np.pad(img, ((ph, ph), (pw, pw)),
                    mode={"zero":"constant","reflect":"reflect",
                          "replicate":"edge","wrap":"wrap"}
                    .get(padding_mode, "reflect"))

    # Stride-trick window view — no Python loops
    out_shape = (H, W, kh, kw)
    strides   = (
        padded.strides[0], padded.strides[1],
        padded.strides[0], padded.strides[1],
    )
    windows = np.lib.stride_tricks.as_strided(padded, shape=out_shape,
                                               strides=strides)
    return (windows * k_flipped).sum(axis=(2, 3)).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 3.5  Spatial filtering  (grayscale + RGB dispatcher)
# ══════════════════════════════════════════════════════════════════════════════

def apply_filter(
    image: np.ndarray,
    kernel: np.ndarray,
    padding_mode: str = "reflect",
) -> np.ndarray:
    """
    Apply a convolution kernel to a grayscale or RGB image.

    For RGB images the kernel is applied independently to each channel
    (per-channel strategy).  For grayscale the call is forwarded directly
    to :func:`convolve2d`.

    Parameters
    ----------
    image : np.ndarray
        Image, shape (H, W) or (H, W, C), any numeric dtype.
        Returned as float32.
    kernel : np.ndarray
        2-D convolution kernel.
    padding_mode : str
        Padding mode.  Default 'reflect'.

    Returns
    -------
    np.ndarray
        Filtered image, same shape as input, dtype float32.

    Raises
    ------
    TypeError / ValueError
        See :func:`convolve2d`.

    Examples
    --------
    >>> k   = np.ones((5, 5), dtype=np.float32) / 25
    >>> out = apply_filter(rgb_img, k)
    """
    _validate_image(image)
    _validate_kernel(kernel)
    img = image.astype(np.float32)
    if img.ndim == 2:
        return convolve2d(img, kernel, padding_mode)
    # 3-D: apply per channel
    channels = [convolve2d(img[:, :, c], kernel, padding_mode)
                for c in range(img.shape[2])]
    return np.stack(channels, axis=2)


# ══════════════════════════════════════════════════════════════════════════════
# 4.1  Mean / Box filter
# ══════════════════════════════════════════════════════════════════════════════

def mean_filter(
    image: np.ndarray,
    kernel_size: int = 3,
    padding_mode: str = "reflect",
) -> np.ndarray:
    """
    Apply a mean (box) filter to an image.

    Every output pixel is the arithmetic mean of the
    ``kernel_size × kernel_size`` neighbourhood centred on it.  This is
    equivalent to convolving with a uniform kernel where every weight is
    ``1 / (kernel_size²)``.

    Math
    ----
    For each pixel (r, c):

        output[r, c] = (1 / k²) · Σ_{i,j ∈ window} image[r+i, c+j]

    where k = kernel_size.

    Parameters
    ----------
    image : np.ndarray
        Input image, shape (H, W) or (H, W, C), any numeric dtype.
    kernel_size : int
        Side length of the square kernel.  Must be a positive odd integer.
        Default 3.
    padding_mode : {'zero', 'reflect', 'replicate', 'wrap'}
        Border handling strategy passed to :func:`pad_image`.
        Default 'reflect'.

    Returns
    -------
    np.ndarray
        Smoothed image, same shape as input, dtype float32, range [0, 1].

    Raises
    ------
    TypeError
        If image is not ndarray, or kernel_size is not int.
    ValueError
        If image shape is invalid, or kernel_size is not a positive odd int.

    Notes
    -----
    * Larger kernel_size → stronger blur but more expensive.
    * Mean filter does not preserve edges well — use Gaussian or median
      for edge-preserving smoothing.

    Examples
    --------
    >>> blurred = mean_filter(img, kernel_size=5)
    """
    _validate_image(image)
    _validate_kernel_size(kernel_size)
    k = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size ** 2)
    return apply_filter(_to_float32(image), k, padding_mode)


# ══════════════════════════════════════════════════════════════════════════════
# 4.2  Gaussian filter
# ══════════════════════════════════════════════════════════════════════════════

def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    """
    Generate a normalised 2-D Gaussian kernel.

    The kernel is computed as the outer product of two 1-D Gaussians:

        G(x, y) = exp(-(x² + y²) / (2σ²))

    then normalised to sum to 1 so energy is preserved.

    Parameters
    ----------
    size : int
        Side length of the square kernel.  Must be a positive odd integer.
        A good rule of thumb: size = 2 * ceil(3 * sigma) + 1.
    sigma : float
        Standard deviation of the Gaussian.  Must be > 0.
        Larger sigma → wider, stronger blur.

    Returns
    -------
    np.ndarray
        2-D Gaussian kernel, shape (size, size), dtype float32,
        values in (0, 1], summing to exactly 1.0.

    Raises
    ------
    TypeError
        If size is not int or sigma is not numeric.
    ValueError
        If size is not a positive odd integer, or sigma <= 0.

    Notes
    -----
    * For sigma → 0 the kernel approaches a delta (identity).
    * The kernel is separable: G(x,y) = G(x) · G(y).  This is exploited
      in :func:`gaussian_filter` for efficiency.

    Examples
    --------
    >>> k = gaussian_kernel(size=5, sigma=1.0)
    >>> k.sum()
    1.0
    """
    _validate_kernel_size(size, "size")
    if not isinstance(sigma, (int, float)):
        raise TypeError(f"'sigma' must be numeric, got {type(sigma).__name__}.")
    if sigma <= 0:
        raise ValueError(f"'sigma' must be > 0, got {sigma}.")

    center = size // 2
    ax     = np.arange(size, dtype=np.float32) - center
    g1d    = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    kernel = np.outer(g1d, g1d).astype(np.float32)
    return kernel / kernel.sum()


def gaussian_filter(
    image: np.ndarray,
    size: int = 5,
    sigma: float = 1.0,
    padding_mode: str = "reflect",
) -> np.ndarray:
    """
    Apply Gaussian smoothing to an image.

    Uses separable convolution — two 1-D passes (row then column) instead
    of one 2-D pass.  This reduces complexity from O(H·W·k²) to O(H·W·k),
    which matters for large kernels.

    Algorithm
    ---------
    1. Generate a 1-D Gaussian kernel of *size* and *sigma*.
    2. Convolve image rows with the 1-D kernel.
    3. Convolve result columns with the same 1-D kernel.
    4. Repeat per channel for RGB images.

    Parameters
    ----------
    image : np.ndarray
        Input image, shape (H, W) or (H, W, C), any numeric dtype.
    size : int
        Kernel side length (odd).  Default 5.
        Recommended: ``2 * ceil(3 * sigma) + 1``.
    sigma : float
        Gaussian standard deviation.  Default 1.0.
    padding_mode : str
        Border handling.  Default 'reflect'.

    Returns
    -------
    np.ndarray
        Smoothed image, same shape, dtype float32, range [0, 1].

    Raises
    ------
    TypeError / ValueError
        See :func:`gaussian_kernel`.

    Notes
    -----
    * Uses the full 2-D kernel for the docstring example but internally
      applies two 1-D passes for performance.
    * Preferred over mean_filter when edge preservation matters less than
      smooth gradients (e.g., before Sobel or Canny).

    Examples
    --------
    >>> blurred = gaussian_filter(img, size=7, sigma=1.5)
    """
    _validate_image(image)
    k2d = gaussian_kernel(size, sigma)
    # Separable: extract 1-D kernel from first row
    k1d = k2d[size // 2, :].copy()
    k1d = (k1d / k1d.sum()).astype(np.float32)

    img = _to_float32(image)

    def _apply_1d(channel: np.ndarray) -> np.ndarray:
        # Row pass
        kr = k1d[np.newaxis, :]
        kr = kr.reshape(1, size)
        row_result = convolve2d(channel, k1d.reshape(1, size), padding_mode)
        # Column pass
        col_result = convolve2d(row_result, k1d.reshape(size, 1), padding_mode)
        return col_result

    if img.ndim == 2:
        return _apply_1d(img)
    return np.stack([_apply_1d(img[:, :, c]) for c in range(img.shape[2])], axis=2)


# ══════════════════════════════════════════════════════════════════════════════
# 4.3  Median filter
# ══════════════════════════════════════════════════════════════════════════════

def median_filter(
    image: np.ndarray,
    kernel_size: int = 3,
    padding_mode: str = "reflect",
) -> np.ndarray:
    """
    Apply a median filter to an image.

    Each output pixel is the **median** of the values in the
    ``kernel_size × kernel_size`` neighbourhood.  The median operation is
    non-linear and cannot be expressed as a convolution, so it requires
    iterating over pixels.

    Loop justification
    ------------------
    The median is inherently a sorting operation over a window of pixels.
    Unlike linear filters (mean, Gaussian), it cannot be written as a
    dot-product of a kernel with a window, so there is no vectorised
    equivalent.  The implementation uses **stride tricks** to extract all
    windows simultaneously into a 4-D array, then calls ``np.median``
    along the window axes — reducing the per-pixel Python loop to a single
    NumPy call.  This is the standard NumPy-only approach and avoids
    pixel-by-pixel Python loops entirely.

    Parameters
    ----------
    image : np.ndarray
        Input image, shape (H, W) or (H, W, C), any numeric dtype.
    kernel_size : int
        Side length of the median window.  Must be a positive odd integer.
        Default 3.
    padding_mode : str
        Border padding mode.  Default 'reflect'.

    Returns
    -------
    np.ndarray
        Filtered image, same shape as input, dtype float32, range [0, 1].

    Raises
    ------
    TypeError
        If image is not ndarray or kernel_size is not int.
    ValueError
        If image shape is invalid, or kernel_size is not a positive odd int.

    Notes
    -----
    * Excellent at removing salt-and-pepper noise while preserving edges.
    * Memory cost: O(H · W · k²) for the window view.  For k > 15 on
      large images, consider tiling.
    * Applied per-channel for RGB images (same as Gaussian).

    Examples
    --------
    >>> clean = median_filter(noisy_img, kernel_size=5)
    """
    _validate_image(image)
    _validate_kernel_size(kernel_size)

    img = _to_float32(image)
    k   = kernel_size
    pad = k // 2

    def _median_channel(channel: np.ndarray) -> np.ndarray:
        H, W = channel.shape
        padded = pad_image(channel, pad, padding_mode)

        # Build 4-D window view — all (H×W) windows of size (k×k) at once
        # Shape: (H, W, k, k) — no Python loops over pixels
        shape   = (H, W, k, k)
        strides = (
            padded.strides[0], padded.strides[1],
            padded.strides[0], padded.strides[1],
        )
        windows = np.lib.stride_tricks.as_strided(padded, shape=shape,
                                                   strides=strides)
        # np.median over the last two axes — one vectorised call
        return np.median(windows, axis=(2, 3)).astype(np.float32)

    if img.ndim == 2:
        return _median_channel(img)
    return np.stack([_median_channel(img[:, :, c])
                     for c in range(img.shape[2])], axis=2)
