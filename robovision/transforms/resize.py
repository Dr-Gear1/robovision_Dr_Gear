"""
robovision/transforms/resize.py
================================
Image resizing with two interpolation methods:
    - Nearest-neighbour  (fast, blocky)
    - Bilinear           (smooth, preferred)

Only NumPy is used — no OpenCV, no scipy.
"""

from __future__ import annotations
import numpy as np


# ── helpers ────────────────────────────────────────────────────────────────

def _validate(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be np.ndarray, got {type(image).__name__}.")
    if image.ndim not in (2, 3):
        raise ValueError(f"image must be 2D or 3D, got shape {image.shape}.")


def _parse_size(new_size: tuple) -> tuple[int, int]:
    if (not isinstance(new_size, (tuple, list))) or len(new_size) != 2:
        raise TypeError("new_size must be a (height, width) tuple.")
    h, w = int(new_size[0]), int(new_size[1])
    if h < 1 or w < 1:
        raise ValueError(f"new_size values must be >= 1, got ({h}, {w}).")
    return h, w


# ── 5.1a  Nearest-neighbour resize ────────────────────────────────────────

def resize_nearest(image: np.ndarray, new_size: tuple) -> np.ndarray:
    """
    Resize an image using nearest-neighbour interpolation.

    Each output pixel is assigned the value of the closest input pixel.
    Fast and simple, but produces a blocky / pixelated result when
    up-scaling.

    Algorithm
    ---------
    For every output coordinate (r', c'):
        r_src = round( r' * H_in / H_out )
        c_src = round( c' * W_in / W_out )
    Then clip to valid range and index directly.

    Parameters
    ----------
    image : np.ndarray
        Input image, shape (H, W) or (H, W, C), any dtype.
    new_size : tuple of int
        Target size as (new_height, new_width).

    Returns
    -------
    np.ndarray
        Resized image, same dtype as input.

    Raises
    ------
    TypeError
        If image is not ndarray or new_size is not a tuple.
    ValueError
        If image is not 2D/3D or new_size values are < 1.

    Examples
    --------
    >>> out = resize_nearest(img, (256, 256))
    """
    _validate(image)
    H_out, W_out = _parse_size(new_size)
    H_in,  W_in  = image.shape[:2]

    # Build coordinate grids for the output
    row_idx = np.round(np.arange(H_out) * H_in / H_out).astype(int)
    col_idx = np.round(np.arange(W_out) * W_in  / W_out).astype(int)

    # Clip to valid input range
    row_idx = np.clip(row_idx, 0, H_in - 1)
    col_idx = np.clip(col_idx, 0, W_in  - 1)

    # Fancy-index in one shot — fully vectorised, no loops
    return image[np.ix_(row_idx, col_idx)]


# ── 5.1b  Bilinear resize ─────────────────────────────────────────────────

def resize_bilinear(image: np.ndarray, new_size: tuple) -> np.ndarray:
    """
    Resize an image using bilinear interpolation.

    Each output pixel is a weighted average of the four surrounding
    input pixels.  Produces smooth results when up-scaling.

    Algorithm
    ---------
    For every output coordinate (r', c'):
        r_src = r' * (H_in - 1) / (H_out - 1)
        c_src = c' * (W_in - 1) / (W_out - 1)

    Let r0, c0 = floor(r_src, c_src),  r1 = r0+1,  c1 = c0+1,
        dr = r_src - r0,  dc = c_src - c0.

    Bilinear formula:
        out = (1-dr)(1-dc)*I[r0,c0] + (1-dr)*dc*I[r0,c1]
            +    dr *(1-dc)*I[r1,c0] +    dr *dc*I[r1,c1]

    Parameters
    ----------
    image : np.ndarray
        Input image, shape (H, W) or (H, W, C), any dtype.
    new_size : tuple of int
        Target size as (new_height, new_width).

    Returns
    -------
    np.ndarray
        Resized image, dtype float32.

    Raises
    ------
    TypeError
        If image is not ndarray or new_size is not a tuple.
    ValueError
        If image is not 2D/3D or new_size values are < 1.

    Notes
    -----
    * Output dtype is always float32 because the weighted average
      produces non-integer values.
    * For single-pixel edge cases (H_in or W_in == 1), nearest-neighbour
      fallback is used on that axis.

    Examples
    --------
    >>> out = resize_bilinear(img, (256, 256))
    """
    _validate(image)
    H_out, W_out = _parse_size(new_size)
    H_in,  W_in  = image.shape[:2]

    img = image.astype(np.float32)

    # Continuous source coordinates for every output pixel
    r_src = np.arange(H_out, dtype=np.float32) * (H_in - 1) / max(H_out - 1, 1)
    c_src = np.arange(W_out, dtype=np.float32) * (W_in  - 1) / max(W_out - 1, 1)

    # Four surrounding integer coordinates
    r0 = np.clip(np.floor(r_src).astype(int), 0, H_in - 1)
    r1 = np.clip(r0 + 1,                      0, H_in - 1)
    c0 = np.clip(np.floor(c_src).astype(int), 0, W_in  - 1)
    c1 = np.clip(c0 + 1,                      0, W_in  - 1)

    # Fractional distances
    dr = (r_src - r0).astype(np.float32)   # shape (H_out,)
    dc = (c_src - c0).astype(np.float32)   # shape (W_out,)

    # Expand to 2D grids — vectorised over both axes simultaneously
    # Add extra dim for channel axis when image is 3D (H,W,C)
    if img.ndim == 3:
        dr = dr[:, np.newaxis, np.newaxis]   # (H_out, 1, 1)
        dc = dc[np.newaxis, :, np.newaxis]   # (1, W_out, 1)
    else:
        dr = dr[:, np.newaxis]               # (H_out, 1)
        dc = dc[np.newaxis, :]               # (1, W_out)

    # Bilinear blend — fully vectorised, works for both 2D and 3D images
    out = (
        (1 - dr) * (1 - dc) * img[np.ix_(r0, c0)] +
        (1 - dr) *      dc  * img[np.ix_(r0, c1)] +
             dr  * (1 - dc) * img[np.ix_(r1, c0)] +
             dr  *      dc  * img[np.ix_(r1, c1)]
    )

    return out.astype(np.float32)


# ── unified entry point ────────────────────────────────────────────────────

def resize(
    image: np.ndarray,
    new_size: tuple,
    method: str = "bilinear",
) -> np.ndarray:
    """
    Resize an image using a chosen interpolation method.

    Parameters
    ----------
    image : np.ndarray
        Input image, shape (H, W) or (H, W, C).
    new_size : tuple of int
        Target (height, width).
    method : {'bilinear', 'nearest'}
        Interpolation method.  Default is 'bilinear'.

    Returns
    -------
    np.ndarray
        Resized image.

    Raises
    ------
    ValueError
        If method is not 'bilinear' or 'nearest'.

    Examples
    --------
    >>> small = resize(img, (128, 128), method='nearest')
    >>> large = resize(img, (512, 512), method='bilinear')
    """
    if method == "bilinear":
        return resize_bilinear(image, new_size)
    elif method == "nearest":
        return resize_nearest(image, new_size)
    else:
        raise ValueError(f"method must be 'bilinear' or 'nearest', got '{method}'.")
