"""
robovision/transforms/rotate.py
================================
Rotate an image about its centre by an arbitrary angle.

Uses inverse mapping so every output pixel is filled (no holes).
Supports nearest-neighbour and bilinear interpolation.

Only NumPy — no OpenCV, no scipy.
"""

from __future__ import annotations
import numpy as np


# ── helpers ────────────────────────────────────────────────────────────────

def _validate(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be np.ndarray, got {type(image).__name__}.")
    if image.ndim not in (2, 3):
        raise ValueError(f"image must be 2D or 3D, got shape {image.shape}.")


def _bilinear_sample(img: np.ndarray, r_src: np.ndarray, c_src: np.ndarray) -> np.ndarray:
    """
    Sample *img* at continuous coordinates (r_src, c_src) using bilinear
    interpolation.  Out-of-bounds coordinates return 0.

    Parameters
    ----------
    img : np.ndarray  shape (H, W) or (H, W, C), float32
    r_src, c_src : np.ndarray  shape (H_out, W_out), continuous coords

    Returns
    -------
    np.ndarray  shape (H_out, W_out) or (H_out, W_out, C), float32
    """
    H, W = img.shape[:2]

    # Boolean mask for in-bounds pixels
    valid = (r_src >= 0) & (r_src <= H - 1) & (c_src >= 0) & (c_src <= W - 1)

    r_src = np.clip(r_src, 0, H - 1)
    c_src = np.clip(c_src, 0, W - 1)

    r0 = np.floor(r_src).astype(int)
    r1 = np.clip(r0 + 1, 0, H - 1)
    c0 = np.floor(c_src).astype(int)
    c1 = np.clip(c0 + 1, 0, W - 1)

    dr = (r_src - r0)[..., np.newaxis] if img.ndim == 3 else (r_src - r0)
    dc = (c_src - c0)[..., np.newaxis] if img.ndim == 3 else (c_src - c0)

    out = (
        (1 - dr) * (1 - dc) * img[r0, c0] +
        (1 - dr) *      dc  * img[r0, c1] +
             dr  * (1 - dc) * img[r1, c0] +
             dr  *      dc  * img[r1, c1]
    ).astype(np.float32)

    # Zero-out pixels that came from outside the source image
    if img.ndim == 3:
        out[~valid] = 0.0
    else:
        out[~valid] = 0.0

    return out


# ── 5.2  Rotate ────────────────────────────────────────────────────────────

def rotate(
    image: np.ndarray,
    angle: float,
    method: str = "bilinear",
    expand: bool = False,
) -> np.ndarray:
    """
    Rotate an image about its centre by *angle* degrees (counter-clockwise).

    Uses **inverse mapping**: for each output pixel we compute where it
    came from in the input, then sample that location.  This guarantees
    every output pixel is filled with no holes.

    Inverse rotation formula
    ------------------------
    Let (cx, cy) be the image centre.  For output pixel (r', c'):

        Δr = r' - cy,   Δc = c' - cx

        r_src =  cos(θ)·Δr + sin(θ)·Δc + cy
        c_src = -sin(θ)·Δr + cos(θ)·Δc + cx

    Parameters
    ----------
    image : np.ndarray
        Input image, shape (H, W) or (H, W, C), any dtype.
    angle : float
        Counter-clockwise rotation angle in degrees.
    method : {'bilinear', 'nearest'}
        Interpolation method.  Default 'bilinear'.
    expand : bool
        If True, expand the output canvas so the full rotated image fits
        without cropping.  If False (default), keep original canvas size.

    Returns
    -------
    np.ndarray
        Rotated image.  dtype float32 for bilinear, same dtype for nearest.
        Pixels outside the original image are filled with 0 (black).

    Raises
    ------
    TypeError
        If image is not ndarray.
    ValueError
        If method is invalid or image has wrong number of dimensions.

    Notes
    -----
    * angle=0 returns a copy of the input (no rotation).
    * angle=90 rotates counter-clockwise by 90 degrees.

    Examples
    --------
    >>> rotated = rotate(img, 45)
    >>> rotated_nn = rotate(img, 30, method='nearest')
    >>> rotated_full = rotate(img, 45, expand=True)
    """
    _validate(image)
    if method not in ("bilinear", "nearest"):
        raise ValueError(f"method must be 'bilinear' or 'nearest', got '{method}'.")

    H_in, W_in = image.shape[:2]
    theta = np.deg2rad(angle)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    if expand:
        # New canvas size that fits the rotated image completely
        corners = np.array([
            [0,     0    ],
            [W_in,  0    ],
            [0,     H_in ],
            [W_in,  H_in ],
        ], dtype=np.float32)
        cx_in = W_in / 2.0
        cy_in = H_in / 2.0
        rotated_corners = np.column_stack([
            cos_t * (corners[:, 0] - cx_in) - sin_t * (corners[:, 1] - cy_in),
            sin_t * (corners[:, 0] - cx_in) + cos_t * (corners[:, 1] - cy_in),
        ])
        W_out = int(np.ceil(rotated_corners[:, 0].max() - rotated_corners[:, 0].min()))
        H_out = int(np.ceil(rotated_corners[:, 1].max() - rotated_corners[:, 1].min()))
    else:
        H_out, W_out = H_in, W_in

    cx_in  = (W_in  - 1) / 2.0
    cy_in  = (H_in  - 1) / 2.0
    cx_out = (W_out - 1) / 2.0
    cy_out = (H_out - 1) / 2.0

    # Build output coordinate grid — shape (H_out, W_out)
    r_out, c_out = np.meshgrid(
        np.arange(H_out, dtype=np.float32),
        np.arange(W_out, dtype=np.float32),
        indexing="ij",
    )

    # Shift to centre
    dr = r_out - cy_out
    dc = c_out - cx_out

    # Inverse rotation (rotate back to find source)
    r_src = cos_t * dr + sin_t * dc + cy_in
    c_src = -sin_t * dr + cos_t * dc + cx_in

    img = image.astype(np.float32)

    if method == "bilinear":
        return _bilinear_sample(img, r_src, c_src)

    else:  # nearest
        r_nn = np.clip(np.round(r_src).astype(int), 0, H_in - 1)
        c_nn = np.clip(np.round(c_src).astype(int), 0, W_in  - 1)
        out  = img[r_nn, c_nn]
        # mask out-of-bounds
        valid = (
            (r_src >= 0) & (r_src <= H_in - 1) &
            (c_src >= 0) & (c_src <= W_in  - 1)
        )
        if img.ndim == 3:
            out[~valid] = 0.0
        else:
            out[~valid] = 0.0
        return out.astype(image.dtype)
