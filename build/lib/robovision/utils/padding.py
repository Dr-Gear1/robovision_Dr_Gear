"""
robovision/utils/padding.py
=============================
Image padding with five border handling strategies.

Public API
----------
pad_image    — unified padding function (5 modes)
unpad_image  — strip padding from a padded image

Padding modes
-------------
+-------------+--------------------------------------------------------------+
| 'zero'      | Fill with 0 (black border). Simplest but causes edge         |
|             | artefacts in filters — high-contrast black ring at borders.  |
+-------------+--------------------------------------------------------------+
| 'reflect'   | Mirror content excluding the edge pixel.                     |
|             | ``|c b| a b c |b c|`` — no duplicate edge pixel.            |
|             | Best default for most filters.                               |
+-------------+--------------------------------------------------------------+
| 'replicate' | Repeat the edge pixel.                                       |
| (edge)      | ``|a a| a b c |c c|``                                        |
|             | Good for gradients near borders.                             |
+-------------+--------------------------------------------------------------+
| 'constant'  | Fill with any user-specified constant value (not just 0).    |
|             | Generalises 'zero'. E.g., fill=0.5 for mid-grey border.      |
+-------------+--------------------------------------------------------------+
| 'circular'  | Wrap-around (periodic / toroidal boundary).                  |
| (wrap)      | ``|b c| a b c |a b|``                                        |
|             | Correct for periodic signals; rarely used in vision.         |
+-------------+--------------------------------------------------------------+

Only NumPy — no OpenCV, no scipy.
"""

from __future__ import annotations
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# Shared validation
# ══════════════════════════════════════════════════════════════════════════════

_VALID_MODES = ("zero", "reflect", "replicate", "constant", "circular")


def _validate(image: np.ndarray, pad: int) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError(
            f"'image' must be numpy.ndarray, got {type(image).__name__}."
        )
    if image.ndim not in (2, 3):
        raise ValueError(
            f"'image' must be 2-D (H, W) or 3-D (H, W, C), "
            f"got shape {image.shape}."
        )
    if image.size == 0:
        raise ValueError("'image' must not be empty.")
    if not isinstance(pad, int):
        raise TypeError(f"'pad_width' must be int, got {type(pad).__name__}.")
    if pad < 0:
        raise ValueError(f"'pad_width' must be >= 0, got {pad}.")


# ══════════════════════════════════════════════════════════════════════════════
# 3.3  Padding (5 modes)
# ══════════════════════════════════════════════════════════════════════════════

def pad_image(
    image: np.ndarray,
    pad_width: int,
    mode: str = "reflect",
    constant_value: float = 0.0,
) -> np.ndarray:
    """
    Pad an image symmetrically on all four sides.

    Adds *pad_width* pixels on each of the four borders (top, bottom,
    left, right) using the specified padding strategy.

    Padding modes — visual examples (pad=2, row=[a,b,c,d,e])
    ---------------------------------------------------------
    +-------------+--------------------------------+
    | zero        | 0 0 | a b c d e | 0 0         |
    +-------------+--------------------------------+
    | reflect     | c b | a b c d e | d c         |
    +-------------+--------------------------------+
    | replicate   | a a | a b c d e | e e         |
    +-------------+--------------------------------+
    | constant    | v v | a b c d e | v v  (v=val)|
    +-------------+--------------------------------+
    | circular    | d e | a b c d e | a b         |
    +-------------+--------------------------------+

    Parameters
    ----------
    image : np.ndarray
        Input image, shape (H, W) or (H, W, C), any dtype.
    pad_width : int
        Number of pixels added to **each** side.  Total size increase:
        height → H + 2*pad_width,  width → W + 2*pad_width.
        Must be >= 0.  pad_width = 0 returns a copy unchanged.
    mode : {'zero', 'reflect', 'replicate', 'constant', 'circular'}
        Padding strategy.  Default 'reflect'.
    constant_value : float, optional
        Fill value used only when mode='constant'.  Default 0.0.
        Ignored for all other modes.

    Returns
    -------
    np.ndarray
        Padded image.
        Shape: (H + 2*pad_width, W + 2*pad_width) or
               (H + 2*pad_width, W + 2*pad_width, C).
        Same dtype as input.

    Raises
    ------
    TypeError
        If image is not ndarray, pad_width is not int, or
        constant_value is not numeric.
    ValueError
        If image shape is invalid, image is empty, pad_width < 0,
        or mode is not one of the five accepted values.

    Notes
    -----
    * pad_width = kernel_size // 2 gives 'same' output after convolution.
    * 'reflect' mode requires pad_width < min(H, W).
    * For 3-D images the channel axis is never padded.

    Examples
    --------
    >>> p = pad_image(img, pad_width=3)                        # reflect
    >>> p = pad_image(img, pad_width=5, mode='zero')
    >>> p = pad_image(img, pad_width=2, mode='constant', constant_value=128)
    >>> p = pad_image(img, pad_width=4, mode='circular')
    >>> p = pad_image(img, pad_width=3, mode='replicate')
    """
    _validate(image, pad_width)
    if mode not in _VALID_MODES:
        raise ValueError(
            f"'mode' must be one of {_VALID_MODES}, got '{mode}'."
        )
    if not isinstance(constant_value, (int, float)):
        raise TypeError(
            f"'constant_value' must be numeric, got {type(constant_value).__name__}."
        )

    if pad_width == 0:
        return image.copy()

    # Build numpy pad spec — channel axis NOT padded for 3-D images
    if image.ndim == 2:
        pad_spec = ((pad_width, pad_width), (pad_width, pad_width))
    else:
        pad_spec = ((pad_width, pad_width), (pad_width, pad_width), (0, 0))

    # Map RoboVision mode names → NumPy mode names
    _NP_MODE = {
        "zero":      ("constant",  {"constant_values": 0}),
        "reflect":   ("reflect",   {}),
        "replicate": ("edge",      {}),
        "constant":  ("constant",  {"constant_values": constant_value}),
        "circular":  ("wrap",      {}),
    }
    np_mode, kwargs = _NP_MODE[mode]

    return np.pad(image, pad_spec, mode=np_mode, **kwargs)


# ══════════════════════════════════════════════════════════════════════════════
# Utility: strip padding
# ══════════════════════════════════════════════════════════════════════════════

def unpad_image(
    padded: np.ndarray,
    pad_width: int,
) -> np.ndarray:
    """
    Strip symmetric padding from an image.

    Removes *pad_width* pixels from all four borders, returning the
    original spatial dimensions.  Inverse of :func:`pad_image`.

    Parameters
    ----------
    padded : np.ndarray
        Padded image, shape (H + 2p, W + 2p) or (H + 2p, W + 2p, C).
    pad_width : int
        Number of pixels to remove from each side.  Must be > 0 and
        smaller than half the spatial dimensions.

    Returns
    -------
    np.ndarray
        Unpadded image, shape (H, W) or (H, W, C).  View of input
        (no copy).

    Raises
    ------
    TypeError
        If padded is not ndarray or pad_width is not int.
    ValueError
        If pad_width <= 0 or image would become empty after stripping.

    Examples
    --------
    >>> original_size = unpad_image(padded, pad_width=3)
    """
    if not isinstance(padded, np.ndarray):
        raise TypeError(
            f"'padded' must be numpy.ndarray, got {type(padded).__name__}."
        )
    if not isinstance(pad_width, int) or pad_width <= 0:
        raise ValueError(
            f"'pad_width' must be a positive int, got {pad_width}."
        )
    H, W = padded.shape[:2]
    if 2 * pad_width >= H or 2 * pad_width >= W:
        raise ValueError(
            f"'pad_width' ({pad_width}) is too large for image of shape "
            f"{padded.shape[:2]} — would produce empty output."
        )
    return padded[pad_width:-pad_width, pad_width:-pad_width]
