"""
robovision/features/spatial_pyramid.py
========================================
Spatial Pyramid Matching (SPM) histogram descriptor.

Divides the image into increasingly fine grids at multiple levels and
computes a color (or gradient) histogram for each cell.  The pyramidal
structure captures both global and local spatial distribution of features.

Reference: Lazebnik, Schmid & Ponce, CVPR 2006 —
           "Beyond Bags of Features: Spatial Pyramid Matching".

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
    img = image.astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0
    return img


def _channel_histogram(pixels: np.ndarray, n_bins: int) -> np.ndarray:
    """Normalised histogram of pixel values in [0, 1]."""
    counts, _ = np.histogram(pixels, bins=n_bins, range=(0.0, 1.0))
    counts = counts.astype(np.float32)
    total = counts.sum()
    if total > 0:
        counts /= total
    return counts


# ── Spatial Pyramid Histogram ─────────────────────────────────────────────

def extract_spatial_pyramid(
    image: np.ndarray,
    levels: int = 3,
    n_bins: int = 16,
    descriptor: str = "color",
    channels: str = "rgb",
) -> np.ndarray:
    """
    Compute a Spatial Pyramid Matching (SPM) histogram descriptor.

    The image is subdivided at L levels:
        Level 0 → 1×1  grid  (whole image)
        Level 1 → 2×2  grid  (4 cells)
        Level 2 → 4×4  grid  (16 cells)
        Level l → 2^l × 2^l grid

    A histogram is computed for each cell; all histograms are weighted
    and concatenated.  The standard Lazebnik weighting is:
        w[0]  = 1 / 2^L
        w[l]  = 1 / 2^(L-l+1)  for l >= 1

    Parameters
    ----------
    image : np.ndarray
        Input image, shape (H, W) or (H, W, C), float [0,1] or uint8.
    levels : int, optional
        Number of pyramid levels (L+1).  Default 3 (levels 0, 1, 2).
    n_bins : int, optional
        Histogram bins per channel.  Default 16.
    descriptor : {'color', 'gray'}, optional
        'color' — per-channel RGB histograms (default).
        'gray'  — single grayscale histogram per cell.
    channels : {'rgb', 'all'}, optional
        Which channels to use when descriptor='color'.  Default 'rgb'.

    Returns
    -------
    np.ndarray
        1-D SPM feature vector, dtype float32.
        Length = Σ_{l=0}^{L} (2^l)² × n_channels × n_bins

        Example (levels=3, n_bins=16, RGB):
            Level 0: 1   cell  × 3 ch × 16 = 48
            Level 1: 4   cells × 3 ch × 16 = 192
            Level 2: 16  cells × 3 ch × 16 = 768
            Total = 1008

    Raises
    ------
    TypeError
        If image is not ndarray.
    ValueError
        If levels < 1, n_bins < 1, or descriptor/channels invalid.

    Notes
    -----
    * Each cell histogram is independently L1-normalised before weighting,
      so cells from different image regions are comparable.
    * For small images, higher pyramid levels may have cells of only 1–2
      pixels — consider keeping levels ≤ 3.

    Examples
    --------
    >>> feat = extract_spatial_pyramid(img, levels=3, n_bins=16)
    >>> feat.shape
    (1008,)   # for RGB image, 3 levels, 16 bins
    """
    if levels < 1:
        raise ValueError(f"levels must be >= 1, got {levels}.")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}.")
    if descriptor not in ("color", "gray"):
        raise ValueError(f"descriptor must be 'color' or 'gray', got '{descriptor}'.")

    img = _validate(image)
    H, W = img.shape[:2]
    L = levels - 1  # maximum level index

    # Determine channel list
    if descriptor == "gray" or img.ndim == 2:
        if img.ndim == 3:
            w    = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)
            gray = (img[:, :, :3] * w).sum(axis=2)
            channel_imgs = [gray]
        else:
            channel_imgs = [img]
    else:
        n_ch = min(3, img.shape[2]) if channels == "rgb" else img.shape[2]
        channel_imgs = [img[:, :, c] for c in range(n_ch)]

    all_features = []

    for level in range(levels):
        n_cells   = 2 ** level
        weight    = 1.0 / (2 ** L) if level == 0 else 1.0 / (2 ** (L - level + 1))

        # Cell boundaries — vectorised using linspace
        row_edges = np.linspace(0, H, n_cells + 1).astype(int)
        col_edges = np.linspace(0, W, n_cells + 1).astype(int)

        for r in range(n_cells):
            for c in range(n_cells):
                r0, r1 = row_edges[r], row_edges[r + 1]
                c0, c1 = col_edges[c], col_edges[c + 1]

                cell_feats = []
                for ch_img in channel_imgs:
                    cell_pixels = ch_img[r0:r1, c0:c1].ravel()
                    h = _channel_histogram(cell_pixels, n_bins)
                    cell_feats.append(h * weight)

                all_features.append(np.concatenate(cell_feats))

    return np.concatenate(all_features).astype(np.float32)


def extract_spatial_pyramid_gradient(
    image: np.ndarray,
    levels: int = 3,
    n_bins: int = 8,
) -> np.ndarray:
    """
    Spatial Pyramid Histogram using gradient orientations instead of color.

    Each cell contains an orientation histogram of the local gradients,
    giving a more shape-aware spatial descriptor than color SPM.

    Parameters
    ----------
    image : np.ndarray
        Input image, shape (H, W) or (H, W, C).
    levels : int, optional
        Pyramid levels.  Default 3.
    n_bins : int, optional
        Orientation histogram bins (unsigned, 0–180°).  Default 8.

    Returns
    -------
    np.ndarray
        1-D SPM gradient feature vector.
        Length = Σ_{l=0}^{L} (2^l)² × n_bins

    Examples
    --------
    >>> feat = extract_spatial_pyramid_gradient(img, levels=3, n_bins=8)
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be np.ndarray, got {type(image).__name__}.")
    img = image.astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0

    # Convert to grayscale
    if img.ndim == 3:
        w    = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)
        gray = (img[:, :, :3] * w).sum(axis=2)
    else:
        gray = img

    # Gradients
    Gx = np.zeros_like(gray)
    Gy = np.zeros_like(gray)
    Gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    Gy[1:-1, :] = gray[2:, :] - gray[:-2, :]

    magnitude   = np.sqrt(Gx ** 2 + Gy ** 2)
    orientation = np.rad2deg(np.arctan2(Gy, Gx)) % 180.0   # unsigned

    H, W = gray.shape
    L    = levels - 1
    all_features = []

    for level in range(levels):
        n_cells  = 2 ** level
        weight   = 1.0 / (2 ** L) if level == 0 else 1.0 / (2 ** (L - level + 1))
        row_edges = np.linspace(0, H, n_cells + 1).astype(int)
        col_edges = np.linspace(0, W, n_cells + 1).astype(int)

        for r in range(n_cells):
            for c in range(n_cells):
                r0, r1 = row_edges[r], row_edges[r + 1]
                c0, c1 = col_edges[c], col_edges[c + 1]

                cell_mag = magnitude[r0:r1, c0:c1].ravel()
                cell_ori = orientation[r0:r1, c0:c1].ravel()

                hist, _ = np.histogram(
                    cell_ori,
                    bins=n_bins,
                    range=(0.0, 180.0),
                    weights=cell_mag,
                )
                hist = hist.astype(np.float32)
                total = hist.sum()
                if total > 0:
                    hist /= total
                all_features.append(hist * weight)

    return np.concatenate(all_features).astype(np.float32)
