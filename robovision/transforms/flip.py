"""
robovision/transforms/flip.py
==============================
Flip an image horizontally, vertically, or both.

Implemented as thin wrappers around NumPy slicing — O(1) views,
no data copied until the caller writes to the array.

Only NumPy — no OpenCV, no scipy.
"""

from __future__ import annotations
import numpy as np


def _validate(image: np.ndarray) -> None:
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be np.ndarray, got {type(image).__name__}.")
    if image.ndim not in (2, 3):
        raise ValueError(f"image must be 2D or 3D, got shape {image.shape}.")


def flip_horizontal(image: np.ndarray) -> np.ndarray:
    """
    Flip an image left-to-right (mirror along vertical axis).

    Parameters
    ----------
    image : np.ndarray
        Input image, shape (H, W) or (H, W, C), any dtype.

    Returns
    -------
    np.ndarray
        Horizontally flipped image, same shape and dtype.
        Returns a copy (not a view) to avoid surprising aliasing.

    Raises
    ------
    TypeError
        If image is not ndarray.
    ValueError
        If image has wrong number of dimensions.

    Examples
    --------
    >>> mirrored = flip_horizontal(img)
    """
    _validate(image)
    return image[:, ::-1].copy()


def flip_vertical(image: np.ndarray) -> np.ndarray:
    """
    Flip an image top-to-bottom (mirror along horizontal axis).

    Parameters
    ----------
    image : np.ndarray
        Input image, shape (H, W) or (H, W, C), any dtype.

    Returns
    -------
    np.ndarray
        Vertically flipped image, same shape and dtype.

    Raises
    ------
    TypeError
        If image is not ndarray.
    ValueError
        If image has wrong number of dimensions.

    Examples
    --------
    >>> flipped = flip_vertical(img)
    """
    _validate(image)
    return image[::-1, :].copy()


def flip_both(image: np.ndarray) -> np.ndarray:
    """
    Flip an image both horizontally and vertically (180-degree rotation).

    Equivalent to ``flip_vertical(flip_horizontal(img))`` but done in
    one slice operation.

    Parameters
    ----------
    image : np.ndarray
        Input image, shape (H, W) or (H, W, C), any dtype.

    Returns
    -------
    np.ndarray
        Image flipped along both axes, same shape and dtype.

    Raises
    ------
    TypeError
        If image is not ndarray.
    ValueError
        If image has wrong number of dimensions.

    Examples
    --------
    >>> rotated_180 = flip_both(img)
    """
    _validate(image)
    return image[::-1, ::-1].copy()


def flip(image: np.ndarray, mode: str = "horizontal") -> np.ndarray:
    """
    Flip an image along the specified axis.

    Parameters
    ----------
    image : np.ndarray
        Input image, shape (H, W) or (H, W, C), any dtype.
    mode : {'horizontal', 'vertical', 'both'}
        Flip direction.  Default is 'horizontal'.

    Returns
    -------
    np.ndarray
        Flipped image, same shape and dtype as input.

    Raises
    ------
    TypeError
        If image is not ndarray.
    ValueError
        If mode is not one of the accepted strings, or image has
        wrong number of dimensions.

    Examples
    --------
    >>> flip(img, mode='horizontal')
    >>> flip(img, mode='vertical')
    >>> flip(img, mode='both')
    """
    if mode == "horizontal":
        return flip_horizontal(image)
    elif mode == "vertical":
        return flip_vertical(image)
    elif mode == "both":
        return flip_both(image)
    else:
        raise ValueError(
            f"mode must be 'horizontal', 'vertical', or 'both', got '{mode}'."
        )
