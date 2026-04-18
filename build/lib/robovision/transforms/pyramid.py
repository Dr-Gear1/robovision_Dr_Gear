"""
robovision/transforms/pyramid.py
==================================
Image pyramids — multi-scale image representations.

Gaussian pyramid  : successive down-sampled + smoothed versions.
Laplacian pyramid : difference between pyramid levels — captures detail.

Only NumPy — no OpenCV, no scipy.
"""

from __future__ import annotations
import numpy as np


# ── internal Gaussian kernel ───────────────────────────────────────────────

def _gaussian_kernel_1d(sigma: float = 1.0, size: int = 5) -> np.ndarray:
    """1-D Gaussian kernel, normalised to sum = 1."""
    if size % 2 == 0:
        size += 1
    center = size // 2
    x = np.arange(size) - center
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def _separable_blur(image: np.ndarray, sigma: float = 1.0, size: int = 5) -> np.ndarray:
    """
    Fast separable Gaussian blur via two 1-D convolutions.
    Works on 2D and 3D (H,W,C) images.
    """
    k = _gaussian_kernel_1d(sigma, size)
    pad = size // 2

    def convolve_1d(img, k, axis):
        """Apply 1-D kernel along one axis using stride tricks (vectorised)."""
        H, W = img.shape[:2]
        if axis == 0:   # along rows
            padded = np.pad(img, ((pad, pad), (0, 0), (0, 0)) if img.ndim == 3
                           else ((pad, pad), (0, 0)), mode='reflect')
            out = sum(k[i] * padded[i:i + H] for i in range(size))
        else:           # along columns
            padded = np.pad(img, ((0, 0), (pad, pad), (0, 0)) if img.ndim == 3
                           else ((0, 0), (pad, pad)), mode='reflect')
            out = sum(k[i] * padded[:, i:i + W] for i in range(size))
        return out.astype(np.float32)

    blurred = convolve_1d(image.astype(np.float32), k, axis=0)
    blurred = convolve_1d(blurred, k, axis=1)
    return blurred


def _validate(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be np.ndarray, got {type(image).__name__}.")
    if image.ndim not in (2, 3):
        raise ValueError(f"image must be 2D or 3D, got shape {image.shape}.")


# ── 5.5a  Gaussian Pyramid ────────────────────────────────────────────────

def gaussian_pyramid(
    image: np.ndarray,
    levels: int = 4,
    sigma: float = 1.0,
) -> list[np.ndarray]:
    """
    Build a Gaussian image pyramid.

    Each level is obtained by blurring the previous level with a
    Gaussian filter and then down-sampling by 2× (keeping every other
    row and column).

    Algorithm
    ---------
    level[0] = original image
    for i in 1 .. levels-1:
        level[i] = downsample( gaussian_blur( level[i-1] ) )

    Parameters
    ----------
    image : np.ndarray
        Input image, shape (H, W) or (H, W, C), any numeric dtype.
    levels : int, optional
        Number of pyramid levels including the original.  Default 4.
        Must be >= 1.
    sigma : float, optional
        Gaussian blur sigma applied at each level.  Default 1.0.

    Returns
    -------
    list of np.ndarray
        List of length *levels*.  Index 0 is the original image;
        each subsequent index is half the spatial resolution of the
        previous one.

    Raises
    ------
    TypeError
        If image is not ndarray.
    ValueError
        If image has wrong dimensions, levels < 1, or the image
        becomes smaller than 1×1 before all levels are built.

    Notes
    -----
    * The pyramid stops early if the image would become smaller than 2×2.
    * All levels are float32.

    Examples
    --------
    >>> pyr = gaussian_pyramid(img, levels=4)
    >>> pyr[0].shape   # original
    (480, 640, 3)
    >>> pyr[1].shape   # half size
    (240, 320, 3)
    """
    _validate(image)
    if not isinstance(levels, int) or levels < 1:
        raise ValueError(f"levels must be an int >= 1, got {levels}.")

    pyramid = [image.astype(np.float32)]
    current = image.astype(np.float32)

    for _ in range(levels - 1):
        H, W = current.shape[:2]
        if H < 2 or W < 2:
            break   # can't downsample further
        blurred   = _separable_blur(current, sigma=sigma)
        current   = blurred[::2, ::2]   # downsample by 2 — vectorised slice
        pyramid.append(current)

    return pyramid


# ── 5.5b  Laplacian Pyramid ───────────────────────────────────────────────

def laplacian_pyramid(
    image: np.ndarray,
    levels: int = 4,
    sigma: float = 1.0,
) -> list[np.ndarray]:
    """
    Build a Laplacian image pyramid.

    Each level captures the detail lost when going from one Gaussian
    level to the next.  The last level is the coarsest Gaussian level.

    Algorithm
    ---------
    gauss = gaussian_pyramid(image, levels)
    for i in 0 .. levels-2:
        upsampled = upsample( gauss[i+1] ) to size of gauss[i]
        lap[i]    = gauss[i] - upsampled
    lap[levels-1] = gauss[levels-1]   # coarsest residual

    The original image can be reconstructed exactly by collapsing:
        for i in levels-2 .. 0:
            gauss[i] = lap[i] + upsample( gauss[i+1] )

    Parameters
    ----------
    image : np.ndarray
        Input image, shape (H, W) or (H, W, C), any numeric dtype.
    levels : int, optional
        Number of pyramid levels.  Default 4.
    sigma : float, optional
        Gaussian sigma used in the underlying Gaussian pyramid.

    Returns
    -------
    list of np.ndarray
        Laplacian pyramid of length *levels*.  Values may be negative
        (difference images).  All levels are float32.

    Raises
    ------
    TypeError / ValueError
        See :func:`gaussian_pyramid`.

    Notes
    -----
    * The pyramid can be collapsed to reconstruct the original:
      use :func:`collapse_laplacian`.

    Examples
    --------
    >>> lap = laplacian_pyramid(img, levels=4)
    >>> lap[0].shape   # same as original — detail at full scale
    (480, 640, 3)
    """
    gauss = gaussian_pyramid(image, levels=levels, sigma=sigma)
    lap   = []

    for i in range(len(gauss) - 1):
        H, W = gauss[i].shape[:2]
        # Upsample gauss[i+1] to match gauss[i] size by repeating pixels
        up = np.repeat(np.repeat(gauss[i + 1], 2, axis=0), 2, axis=1)
        # Crop / pad to exactly match gauss[i] dimensions
        up = up[:H, :W]
        if up.shape[0] < H:
            pad_r = H - up.shape[0]
            up = np.concatenate([up, np.zeros((pad_r, W) + up.shape[2:], dtype=np.float32)], axis=0)
        if up.shape[1] < W:
            pad_c = W - up.shape[1]
            up = np.concatenate([up, np.zeros((H, pad_c) + up.shape[2:], dtype=np.float32)], axis=1)
        lap.append(gauss[i] - up)

    lap.append(gauss[-1])   # coarsest residual level
    return lap


def collapse_laplacian(lap: list[np.ndarray]) -> np.ndarray:
    """
    Reconstruct the original image from a Laplacian pyramid.

    Parameters
    ----------
    lap : list of np.ndarray
        Laplacian pyramid returned by :func:`laplacian_pyramid`.

    Returns
    -------
    np.ndarray
        Reconstructed image, float32.  Should closely match the original
        (minor differences due to float32 precision).

    Examples
    --------
    >>> lap = laplacian_pyramid(img, levels=4)
    >>> reconstructed = collapse_laplacian(lap)
    """
    current = lap[-1].copy()
    for level in reversed(lap[:-1]):
        H, W = level.shape[:2]
        up = np.repeat(np.repeat(current, 2, axis=0), 2, axis=1)
        up = up[:H, :W]
        current = level + up
    return current.astype(np.float32)
