"""
robovision/features/color_histogram.py
========================================
Color histogram feature descriptor.

Computes per-channel histograms and concatenates them into a single
feature vector.  Works with RGB, RGBA, and grayscale images.

Only NumPy — no OpenCV, no scipy.
"""

from __future__ import annotations
import numpy as np


# ── validation ────────────────────────────────────────────────────────────

def _validate(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be np.ndarray, got {type(image).__name__}.")
    if image.ndim not in (2, 3):
        raise ValueError(f"image must be 2D or 3D, got shape {image.shape}.")
    return image.astype(np.float32)


# ── Color Histogram ───────────────────────────────────────────────────────

def extract_color_histogram(
    image: np.ndarray,
    n_bins: int = 32,
    normalize: bool = True,
    channels: str = "all",
) -> np.ndarray:
    """
    Compute a concatenated per-channel color histogram.

    For an RGB image with n_bins=32:
        R histogram (32) + G histogram (32) + B histogram (32) = 96-D vector.
    For a grayscale image:
        single histogram = n_bins-D vector.

    Algorithm
    ---------
    1. For each channel, bin pixel values into [0, 1] range → n_bins.
    2. Count pixels per bin using np.histogram (vectorised, no loops).
    3. Optionally L1-normalise each channel's histogram.
    4. Concatenate all channel histograms.

    Parameters
    ----------
    image : np.ndarray
        Input image, shape (H, W) or (H, W, C), dtype float32 or uint8.
        Values are expected in [0, 1] for float or [0, 255] for uint8.
    n_bins : int, optional
        Number of bins per channel.  Default 32.
    normalize : bool, optional
        If True (default), normalise each channel histogram to sum to 1.
    channels : {'all', 'rgb', 'gray'}, optional
        'all'  — use all available channels (default).
        'rgb'  — use only first 3 channels (drop alpha if RGBA).
        'gray' — convert to grayscale first, single histogram.

    Returns
    -------
    np.ndarray
        1-D concatenated histogram, dtype float32.
        Length = n_channels × n_bins.

    Raises
    ------
    TypeError
        If image is not ndarray.
    ValueError
        If image dimensions are wrong, n_bins < 1, or channels is invalid.

    Notes
    -----
    * uint8 images [0, 255] are normalised to [0, 1] automatically.
    * Alpha channel is silently dropped when channels='rgb'.

    Examples
    --------
    >>> feat = extract_color_histogram(img, n_bins=32)
    >>> feat.shape
    (96,)   # for RGB image with 32 bins
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}.")
    if channels not in ("all", "rgb", "gray"):
        raise ValueError(f"channels must be 'all', 'rgb', or 'gray', got '{channels}'.")

    img = _validate(image)

    # Normalise uint8 → float [0, 1]
    if img.max() > 1.0:
        img = img / 255.0

    # Select channels
    if channels == "gray" or img.ndim == 2:
        if img.ndim == 3:
            w   = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)
            img = (img[:, :, :3] * w).sum(axis=2)
        channel_list = [img]
    elif channels == "rgb":
        channel_list = [img[:, :, c] for c in range(min(3, img.shape[2]))]
    else:   # all
        n_ch = img.shape[2] if img.ndim == 3 else 1
        channel_list = [img] if img.ndim == 2 else [img[:, :, c] for c in range(n_ch)]

    histograms = []
    for ch in channel_list:
        counts, _ = np.histogram(ch.ravel(), bins=n_bins, range=(0.0, 1.0))
        counts = counts.astype(np.float32)
        if normalize:
            total = counts.sum()
            if total > 0:
                counts /= total
        histograms.append(counts)

    return np.concatenate(histograms).astype(np.float32)


def extract_color_histogram_2d(
    image: np.ndarray,
    n_bins: int = 32,
    channel_pair: tuple[int, int] = (0, 1),
) -> np.ndarray:
    """
    Compute a 2-D joint histogram between two channels.

    Captures the relationship between channels (e.g., R vs G) —
    useful for detecting specific color combinations.

    Parameters
    ----------
    image : np.ndarray
        RGB or RGBA image, shape (H, W, C).
    n_bins : int
        Bins per axis.  Default 32.  Output is (n_bins, n_bins).
    channel_pair : tuple of two ints
        Which two channels to correlate.  Default (0, 1) = R vs G.

    Returns
    -------
    np.ndarray
        2-D histogram, shape (n_bins, n_bins), normalised to sum = 1.
        Flattened to 1-D if you need a feature vector: `hist.ravel()`.

    Raises
    ------
    TypeError / ValueError
        On invalid inputs.

    Examples
    --------
    >>> hist2d = extract_color_histogram_2d(img, n_bins=16)
    >>> feat   = hist2d.ravel()   # 256-D vector
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be np.ndarray, got {type(image).__name__}.")
    if image.ndim != 3 or image.shape[2] < 2:
        raise ValueError("image must be 3D with at least 2 channels for 2D histogram.")
    img = image.astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0
    c1, c2 = channel_pair
    h, _, _ = np.histogram2d(
        img[:, :, c1].ravel(),
        img[:, :, c2].ravel(),
        bins=n_bins,
        range=[[0, 1], [0, 1]],
    )
    h = h.astype(np.float32)
    total = h.sum()
    if total > 0:
        h /= total
    return h
