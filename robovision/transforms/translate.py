"""
robovision/transforms/translate.py
====================================
Shift an image by (tx, ty) pixels along x (columns) and y (rows).

Only NumPy — no OpenCV, no scipy.
"""

from __future__ import annotations
import numpy as np


def _validate(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be np.ndarray, got {type(image).__name__}.")
    if image.ndim not in (2, 3):
        raise ValueError(f"image must be 2D or 3D, got shape {image.shape}.")


def translate(
    image: np.ndarray,
    tx: int,
    ty: int,
    fill: float = 0.0,
) -> np.ndarray:
    """
    Translate (shift) an image by (tx, ty) pixels.

    Positive tx shifts right, negative tx shifts left.
    Positive ty shifts down,  negative ty shifts up.
    Pixels that move outside the canvas are discarded; newly exposed
    pixels are filled with *fill*.

    Algorithm
    ---------
    Build an output canvas of the same size filled with *fill*.
    Compute the source and destination slice ranges for rows and columns,
    then copy in one NumPy slice assignment — no Python loops.

    Parameters
    ----------
    image : np.ndarray
        Input image, shape (H, W) or (H, W, C), any dtype.
    tx : int
        Horizontal shift in pixels.  Positive → right.
    ty : int
        Vertical shift in pixels.   Positive → down.
    fill : float, optional
        Value used to fill newly exposed border pixels.  Default 0.0.

    Returns
    -------
    np.ndarray
        Translated image, same shape and dtype as input.

    Raises
    ------
    TypeError
        If image is not ndarray, or tx / ty are not int-compatible.
    ValueError
        If image has wrong number of dimensions.

    Notes
    -----
    * If |tx| >= W or |ty| >= H the output is entirely *fill* (full shift
      out of frame).
    * dtype is preserved — fill is cast to image.dtype before use.

    Examples
    --------
    >>> shifted_right = translate(img, tx=50,  ty=0)
    >>> shifted_down  = translate(img, tx=0,   ty=30)
    >>> shifted_diag  = translate(img, tx=-20, ty=15)
    """
    _validate(image)
    tx, ty = int(tx), int(ty)

    H, W = image.shape[:2]
    out = np.full_like(image, fill_value=fill, dtype=image.dtype)

    # Source and destination row ranges
    src_r_start = max(0, -ty)
    src_r_end   = min(H,  H - ty)
    dst_r_start = max(0,  ty)
    dst_r_end   = min(H,  H + ty)

    # Source and destination column ranges
    src_c_start = max(0, -tx)
    src_c_end   = min(W,  W - tx)
    dst_c_start = max(0,  tx)
    dst_c_end   = min(W,  W + tx)

    # Guard against full out-of-frame shifts
    if src_r_start >= src_r_end or src_c_start >= src_c_end:
        return out   # entirely out of frame → return blank canvas

    # Single slice assignment — vectorised
    out[dst_r_start:dst_r_end, dst_c_start:dst_c_end] = \
        image[src_r_start:src_r_end, src_c_start:src_c_end]

    return out
