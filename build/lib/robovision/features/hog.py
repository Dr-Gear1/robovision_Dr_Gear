"""
robovision/features/hog.py
===========================
Histogram of Oriented Gradients (HOG) feature descriptor.

HOG captures local shape and texture by computing gradient orientations
in small cells and normalising across larger blocks.

Reference: Dalal & Triggs, CVPR 2005.
Only NumPy — no OpenCV, no scipy.
"""

from __future__ import annotations
import numpy as np


# ── validation ────────────────────────────────────────────────────────────

def _validate(image: np.ndarray) -> np.ndarray:
    """Convert to float32 grayscale (H, W). Accepts 2D or 3D input."""
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be np.ndarray, got {type(image).__name__}.")
    if image.ndim == 3:
        # Luminance conversion (BT.601)
        weights = np.array([0.2989, 0.5870, 0.1140], dtype=np.float64)
        image = (image[:, :, :3].astype(np.float64) * weights).sum(axis=2)
    elif image.ndim != 2:
        raise ValueError(f"image must be 2D or 3D, got shape {image.shape}.")
    return image.astype(np.float32)


# ── HOG ───────────────────────────────────────────────────────────────────

def extract_hog(
    image: np.ndarray,
    cell_size: int = 8,
    block_size: int = 2,
    n_bins: int = 9,
    signed: bool = False,
) -> np.ndarray:
    """
    Compute the HOG descriptor for an image.

    Algorithm
    ---------
    1. Convert to grayscale (if RGB/RGBA).
    2. Compute pixel-wise gradients (Gx, Gy) using [-1, 0, 1] filters.
    3. Compute gradient magnitude and orientation.
    4. Divide image into (cell_size × cell_size) cells; build a soft
       orientation histogram for each cell by voting with magnitude.
    5. Group cells into (block_size × block_size) blocks; L2-normalise
       each block to reduce illumination sensitivity.
    6. Concatenate all normalised block descriptors into one vector.

    Parameters
    ----------
    image : np.ndarray
        Input image, shape (H, W) or (H, W, C), any dtype.
        Converted to float32 grayscale internally.
    cell_size : int, optional
        Width and height of each cell in pixels.  Default 8.
    block_size : int, optional
        Number of cells per block side.  Default 2 (2×2 cells per block).
    n_bins : int, optional
        Number of orientation histogram bins.  Default 9.
    signed : bool, optional
        If False (default), use unsigned gradients [0°, 180°).
        If True, use signed gradients [0°, 360°).

    Returns
    -------
    np.ndarray
        1-D HOG feature vector, dtype float32.
        Length = n_x_blocks * n_y_blocks * block_size² * n_bins
        where n_x_blocks = (W/cell_size - block_size + 1), etc.

    Raises
    ------
    TypeError
        If image is not ndarray.
    ValueError
        If image has wrong dimensions or cell_size < 1.

    Notes
    -----
    * The image is NOT resized internally — pass a fixed-size image if
      you need a fixed-length descriptor across a dataset.
    * Soft bilinear voting between bins is used (each gradient vote is
      split between the two nearest histogram bins).

    Examples
    --------
    >>> feat = extract_hog(img, cell_size=8, block_size=2, n_bins=9)
    >>> feat.shape
    (3780,)   # for a 64×128 image
    """
    if cell_size < 1:
        raise ValueError(f"cell_size must be >= 1, got {cell_size}.")
    gray = _validate(image)
    H, W = gray.shape

    # ── Step 1: Gradients ─────────────────────────────────────────────
    # Central difference in x and y directions — vectorised
    Gx = np.zeros_like(gray)
    Gy = np.zeros_like(gray)
    Gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]   # horizontal gradient
    Gy[1:-1, :] = gray[2:, :] - gray[:-2, :]   # vertical gradient

    magnitude   = np.sqrt(Gx ** 2 + Gy ** 2)
    orientation = np.rad2deg(np.arctan2(Gy, Gx))   # [-180, 180]

    # ── Step 2: Map orientations to bins ──────────────────────────────
    angle_range = 360.0 if signed else 180.0
    orientation = orientation % angle_range          # [0, 360) or [0, 180)

    # ── Step 3: Build per-cell histograms ─────────────────────────────
    n_cells_y = H // cell_size
    n_cells_x = W // cell_size

    # Cell histogram tensor: (n_cells_y, n_cells_x, n_bins)
    cell_hog = np.zeros((n_cells_y, n_cells_x, n_bins), dtype=np.float32)

    bin_width = angle_range / n_bins

    for b in range(n_bins):
        # Soft bilinear voting — each pixel votes for two adjacent bins
        bin_center = (b + 0.5) * bin_width
        diff       = orientation - bin_center
        # Wrap-around distance
        diff       = (diff + angle_range / 2) % angle_range - angle_range / 2
        weight     = np.maximum(0.0, 1.0 - np.abs(diff) / bin_width)
        votes      = magnitude * weight

        # Sum votes inside each cell using reshape + sum (no explicit loops)
        cropped = votes[:n_cells_y * cell_size, :n_cells_x * cell_size]
        cell_hog[:, :, b] = (
            cropped
            .reshape(n_cells_y, cell_size, n_cells_x, cell_size)
            .sum(axis=(1, 3))
        )

    # ── Step 4: Block normalisation (L2-Hys) ──────────────────────────
    n_blocks_y = n_cells_y - block_size + 1
    n_blocks_x = n_cells_x - block_size + 1
    eps = 1e-6

    descriptors = []
    for by in range(n_blocks_y):
        for bx in range(n_blocks_x):
            block = cell_hog[by:by + block_size, bx:bx + block_size, :]
            vec   = block.ravel()
            vec   = vec / np.sqrt(np.dot(vec, vec) + eps ** 2)   # L2 norm
            vec   = np.clip(vec, 0, 0.2)                          # Hys clipping
            vec   = vec / np.sqrt(np.dot(vec, vec) + eps ** 2)   # renormalise
            descriptors.append(vec)

    return np.concatenate(descriptors).astype(np.float32)


def extract_hog_visual(
    image: np.ndarray,
    cell_size: int = 8,
    n_bins: int = 9,
    signed: bool = False,
    scale: float = 2.0,
) -> np.ndarray:
    """
    Produce a visualisation of HOG orientations overlaid on a canvas.

    Each cell is represented by a line segment oriented at the dominant
    gradient direction, with length proportional to the bin magnitude.

    Parameters
    ----------
    image : np.ndarray
        Input image, shape (H, W) or (H, W, C).
    cell_size : int
        Cell size in pixels.  Default 8.
    n_bins : int
        Number of orientation bins.  Default 9.
    signed : bool
        Signed (360°) or unsigned (180°) gradients.  Default False.
    scale : float
        Amplification factor for line lengths.  Default 2.0.

    Returns
    -------
    np.ndarray
        Grayscale visualisation, shape (H, W), dtype float32, range [0, 1].

    Examples
    --------
    >>> vis = extract_hog_visual(img)
    """
    gray = _validate(image)
    H, W = gray.shape
    n_cells_y = H // cell_size
    n_cells_x = W // cell_size
    angle_range = 360.0 if signed else 180.0

    Gx = np.zeros_like(gray)
    Gy = np.zeros_like(gray)
    Gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    Gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    magnitude   = np.sqrt(Gx ** 2 + Gy ** 2)
    orientation = np.rad2deg(np.arctan2(Gy, Gx)) % angle_range
    bin_width   = angle_range / n_bins

    canvas = np.zeros((H, W), dtype=np.float32)
    half   = cell_size // 2

    for cy in range(n_cells_y):
        for cx in range(n_cells_x):
            r0 = cy * cell_size
            c0 = cx * cell_size
            cell_mag = magnitude[r0:r0 + cell_size, c0:c0 + cell_size]
            cell_ori = orientation[r0:r0 + cell_size, c0:c0 + cell_size]

            hist = np.zeros(n_bins)
            for b in range(n_bins):
                bin_center = (b + 0.5) * bin_width
                diff = (cell_ori - bin_center + angle_range / 2) % angle_range - angle_range / 2
                w    = np.maximum(0.0, 1.0 - np.abs(diff) / bin_width)
                hist[b] = (cell_mag * w).sum()

            dominant = np.argmax(hist)
            angle_rad = np.deg2rad((dominant + 0.5) * bin_width)
            dy = int(round(np.sin(angle_rad) * half * scale))
            dx = int(round(np.cos(angle_rad) * half * scale))
            cy_center = r0 + half
            cx_center = c0 + half

            # Draw line using Bresenham-like rasterisation
            for t in np.linspace(-1, 1, cell_size):
                rr = int(cy_center + t * dy)
                cc = int(cx_center + t * dx)
                if 0 <= rr < H and 0 <= cc < W:
                    canvas[rr, cc] = hist[dominant] / (cell_mag.sum() + 1e-6)

    return canvas
