"""
robovision/utils/normalization.py
===================================
Image normalization — three modes covering the full spectrum from
simple range scaling to statistical standardisation.

Public API
----------
normalize_minmax   — scale to [0, 1] or any [a, b] range
normalize_zscore   — zero-mean, unit-variance standardisation
normalize_scale    — scale to [0, 1] or [0, 255] with dtype control
normalize          — unified entry point (dispatch by mode name)

Only NumPy — no OpenCV, no scipy.
"""

from __future__ import annotations
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# Shared validation helpers
# ══════════════════════════════════════════════════════════════════════════════

def _validate_image(image: np.ndarray, name: str = "image") -> None:
    """Raise TypeError / ValueError for invalid image inputs."""
    if not isinstance(image, np.ndarray):
        raise TypeError(
            f"'{name}' must be a numpy.ndarray, "
            f"got {type(image).__name__}."
        )
    if image.ndim not in (2, 3):
        raise ValueError(
            f"'{name}' must be 2-D (H, W) or 3-D (H, W, C), "
            f"got shape {image.shape}."
        )
    if image.size == 0:
        raise ValueError(f"'{name}' must not be empty.")


# ══════════════════════════════════════════════════════════════════════════════
# 3.1 Mode 1 — Min-Max normalization
# ══════════════════════════════════════════════════════════════════════════════

def normalize_minmax(
    image: np.ndarray,
    out_min: float = 0.0,
    out_max: float = 1.0,
) -> np.ndarray:
    """
    Normalise an image using min-max scaling.

    Linearly maps the pixel range [img_min, img_max] to [out_min, out_max].

    Formula
    -------
    For each pixel p:

        p' = (p - img_min) / (img_max - img_min) * (out_max - out_min) + out_min

    If the image is constant (img_min == img_max), all pixels are mapped
    to out_min to avoid division by zero.

    Parameters
    ----------
    image : np.ndarray
        Input image, shape (H, W) or (H, W, C), any numeric dtype.
    out_min : float, optional
        Minimum value of the output range.  Default 0.0.
    out_max : float, optional
        Maximum value of the output range.  Default 1.0.

    Returns
    -------
    np.ndarray
        Normalised image, same shape as input, dtype float32.
        All values are guaranteed to be in [out_min, out_max].

    Raises
    ------
    TypeError
        If image is not ndarray, or out_min / out_max are not numeric.
    ValueError
        If image shape is invalid, image is empty, or out_min >= out_max.

    Notes
    -----
    * Fully vectorised — no Python loops.
    * Does not modify the input array in place.

    Examples
    --------
    >>> norm = normalize_minmax(img)                    # → [0.0, 1.0]
    >>> norm = normalize_minmax(img, out_min=0, out_max=255)  # → [0, 255]
    """
    _validate_image(image)
    for name, val in [("out_min", out_min), ("out_max", out_max)]:
        if not isinstance(val, (int, float)):
            raise TypeError(
                f"'{name}' must be numeric, got {type(val).__name__}."
            )
    if out_min >= out_max:
        raise ValueError(
            f"'out_min' ({out_min}) must be strictly less than "
            f"'out_max' ({out_max})."
        )

    img      = image.astype(np.float32)
    img_min  = img.min()
    img_max  = img.max()
    span_in  = img_max - img_min
    span_out = float(out_max) - float(out_min)

    if span_in == 0:
        # Constant image — map everything to out_min
        return np.full_like(img, fill_value=out_min, dtype=np.float32)

    normalised = (img - img_min) / span_in * span_out + out_min
    return normalised.astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 3.1 Mode 2 — Z-score normalization
# ══════════════════════════════════════════════════════════════════════════════

def normalize_zscore(
    image: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """
    Normalise an image using Z-score (standard score) standardisation.

    Subtracts the global mean and divides by the global standard deviation,
    producing an output with zero mean and unit variance.

    Formula
    -------
        p' = (p - μ) / (σ + ε)

    where μ = mean of all pixels, σ = std of all pixels,
    ε is a small constant to avoid division by zero.

    Parameters
    ----------
    image : np.ndarray
        Input image, shape (H, W) or (H, W, C), any numeric dtype.
    eps : float, optional
        Small constant added to std to prevent division by zero.
        Default 1e-8.

    Returns
    -------
    np.ndarray
        Standardised image, same shape, dtype float32.
        Output is NOT bounded to [0, 1] — values can be negative
        or greater than 1.  Use :func:`normalize_minmax` afterwards
        if a bounded range is needed.

    Raises
    ------
    TypeError
        If image is not ndarray.
    ValueError
        If image shape or size is invalid, or eps <= 0.

    Notes
    -----
    * Z-score is computed over **all channels jointly** (global stats).
      For per-channel normalisation, apply the function channel-by-channel.
    * Preferred before training neural networks where zero-centred inputs
      improve convergence.

    Examples
    --------
    >>> z = normalize_zscore(img)
    >>> print(z.mean(), z.std())   # ≈ 0.0, ≈ 1.0
    """
    _validate_image(image)
    if not isinstance(eps, (int, float)) or eps <= 0:
        raise ValueError(f"'eps' must be a positive number, got {eps}.")

    img = image.astype(np.float32)
    mu  = img.mean()
    sigma = img.std()

    return ((img - mu) / (sigma + eps)).astype(np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# 3.1 Mode 3 — Scale to [0, 1] or [0, 255]
# ══════════════════════════════════════════════════════════════════════════════

def normalize_scale(
    image: np.ndarray,
    target: str = "0-1",
) -> np.ndarray:
    """
    Scale pixel values to a standard range with explicit dtype control.

    Two targets are supported:

    +----------+-------------------------------------------+------------+
    | target   | Output range                              | dtype      |
    +==========+===========================================+============+
    | '0-1'    | [0.0, 1.0] — standard float range        | float32    |
    +----------+-------------------------------------------+------------+
    | '0-255'  | [0, 255] — standard uint8 display range  | uint8      |
    +----------+-------------------------------------------+------------+

    The function auto-detects the input range:
    - If max > 1.0 → treats as [0, 255] uint8 range.
    - If max ≤ 1.0 → treats as [0, 1.0] float range.

    Parameters
    ----------
    image : np.ndarray
        Input image, shape (H, W) or (H, W, C), any numeric dtype.
    target : {'0-1', '0-255'}
        Desired output range.  Default '0-1'.

    Returns
    -------
    np.ndarray
        Scaled image, same shape.
        dtype float32 for target='0-1', uint8 for target='0-255'.

    Raises
    ------
    TypeError
        If image is not ndarray.
    ValueError
        If image shape is invalid or target is not '0-1' / '0-255'.

    Examples
    --------
    >>> f = normalize_scale(uint8_img, target='0-1')     # float32 [0,1]
    >>> u = normalize_scale(float_img, target='0-255')   # uint8  [0,255]
    """
    _validate_image(image)
    if target not in ("0-1", "0-255"):
        raise ValueError(
            f"'target' must be '0-1' or '0-255', got '{target}'."
        )

    img = image.astype(np.float32)
    # Auto-detect input range
    if img.max() > 1.0:
        img = img / 255.0    # normalise to [0,1] first

    img = np.clip(img, 0.0, 1.0)

    if target == "0-1":
        return img.astype(np.float32)
    else:  # 0-255
        return (img * 255.0).round().astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# Unified entry point
# ══════════════════════════════════════════════════════════════════════════════

_MODES = ("minmax", "zscore", "scale_01", "scale_255")


def normalize(
    image: np.ndarray,
    mode: str = "minmax",
    **kwargs,
) -> np.ndarray:
    """
    Unified normalization entry point — dispatches to the correct function.

    Parameters
    ----------
    image : np.ndarray
        Input image, shape (H, W) or (H, W, C).
    mode : {'minmax', 'zscore', 'scale_01', 'scale_255'}
        Normalization method:
        - 'minmax'    → :func:`normalize_minmax`  (default)
        - 'zscore'    → :func:`normalize_zscore`
        - 'scale_01'  → :func:`normalize_scale` with target='0-1'
        - 'scale_255' → :func:`normalize_scale` with target='0-255'
    **kwargs
        Extra arguments forwarded to the chosen function.
        E.g. ``normalize(img, mode='minmax', out_min=0, out_max=255)``.

    Returns
    -------
    np.ndarray
        Normalised image.

    Raises
    ------
    TypeError / ValueError
        See individual function docs.

    Examples
    --------
    >>> normalize(img, mode='minmax')
    >>> normalize(img, mode='zscore')
    >>> normalize(img, mode='scale_01')
    >>> normalize(img, mode='scale_255')
    """
    if mode == "minmax":
        return normalize_minmax(image, **kwargs)
    elif mode == "zscore":
        return normalize_zscore(image, **kwargs)
    elif mode == "scale_01":
        return normalize_scale(image, target="0-1", **kwargs)
    elif mode == "scale_255":
        return normalize_scale(image, target="0-255", **kwargs)
    else:
        raise ValueError(
            f"'mode' must be one of {_MODES}, got '{mode}'."
        )
