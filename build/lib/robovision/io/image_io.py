"""
robovision/io/image_io.py
=========================
Image I/O and color-conversion utilities for the RoboVision library.

Covers project requirements:
    2.1  read_image   – Load an image from disk into a NumPy array.
    2.2  save_image   – Export a NumPy array to PNG / JPG on disk.
    2.3  to_grayscale – Convert an RGB image to grayscale.
         to_rgb       – Convert a grayscale image to a 3-channel RGB array.

Allowed dependencies: NumPy, Matplotlib, Python standard library only.
"""

from __future__ import annotations

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_ndarray(image: np.ndarray, name: str = "image") -> None:
    """Raise TypeError if *image* is not a NumPy ndarray."""
    if not isinstance(image, np.ndarray):
        raise TypeError(
            f"'{name}' must be a NumPy ndarray, got {type(image).__name__}."
        )


def _validate_image_shape(image: np.ndarray, name: str = "image") -> None:
    """
    Raise ValueError if *image* does not have a valid 2-D (H, W) or
    3-D (H, W, C) shape, where C ∈ {1, 3, 4}.
    """
    if image.ndim == 2:
        return  # grayscale – always valid
    if image.ndim == 3 and image.shape[2] in (1, 3, 4):
        return  # single-channel / RGB / RGBA – valid
    raise ValueError(
        f"'{name}' must be shaped (H, W) or (H, W, C) with C ∈ {{1, 3, 4}}, "
        f"got shape {image.shape}."
    )


def _validate_path_string(path: str, name: str = "path") -> None:
    """Raise TypeError if *path* is not a non-empty string."""
    if not isinstance(path, str):
        raise TypeError(
            f"'{name}' must be a str, got {type(path).__name__}."
        )
    if not path.strip():
        raise ValueError(f"'{name}' must not be an empty string.")


# ---------------------------------------------------------------------------
# 2.1  Read image
# ---------------------------------------------------------------------------

def read_image(path: str, as_gray: bool = False) -> np.ndarray:
    """
    Load an image from disk into a NumPy array.

    Uses Matplotlib's image reader, which supports PNG natively and JPG /
    JPEG through Pillow when it is installed as a Matplotlib backend plugin.
    The returned array is always float32 in the range [0.0, 1.0].

    Parameters
    ----------
    path : str
        Absolute or relative path to the image file.
        Supported extensions: .png, .jpg, .jpeg (backend-dependent).
    as_gray : bool, optional
        If True the image is converted to grayscale before returning,
        yielding a 2-D array of shape (H, W).  Default is False.

    Returns
    -------
    np.ndarray
        - RGB  image → shape (H, W, 3),  dtype float32, values in [0, 1].
        - RGBA image → shape (H, W, 4),  dtype float32, values in [0, 1].
        - Gray image → shape (H, W),     dtype float32, values in [0, 1].
          (when *as_gray* is True or the source file is already grayscale)

    Raises
    ------
    TypeError
        If *path* is not a str.
    ValueError
        If *path* is an empty string.
    FileNotFoundError
        If the file does not exist at the given *path*.
    OSError
        If Matplotlib cannot open / decode the file
        (e.g., unsupported format or corrupted file).

    Notes
    -----
    * PNG files with an alpha channel are read as (H, W, 4).
      Use :func:`drop_alpha` if you need a pure RGB array.
    * The function **never** returns uint8; callers that need integer pixel
      values should call ``.astype(np.uint8)`` on the result after scaling
      by 255.

    Examples
    --------
    >>> img = read_image("photo.png")          # RGB  → (H, W, 3) float32
    >>> gray = read_image("photo.png", as_gray=True)  # → (H, W) float32
    """
    _validate_path_string(path, "path")

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No file found at path: '{path}'. "
            "Check spelling and that the file exists."
        )

    # Matplotlib reads PNG as float32 in [0,1]; JPG as uint8 in [0,255].
    raw = mpimg.imread(path)

    # Normalise uint8 images (JPG) to float32 [0, 1]
    if raw.dtype == np.uint8:
        image = raw.astype(np.float32) / 255.0
    elif raw.dtype in (np.float32, np.float64):
        image = raw.astype(np.float32)
    else:
        # Fallback: cast and normalise by the dtype max
        image = (raw.astype(np.float64) / np.iinfo(raw.dtype).max).astype(np.float32)

    if as_gray:
        image = to_grayscale(image)

    return image


# ---------------------------------------------------------------------------
# 2.2  Export / save image
# ---------------------------------------------------------------------------

def save_image(
    image: np.ndarray,
    path: str,
    cmap: str | None = None,
    quality: int = 95,
) -> None:
    """
    Export a NumPy array to a PNG or JPG image file on disk.

    Accepts both grayscale (H, W) and RGB / RGBA (H, W, C) arrays.
    The pixel-value range may be either [0, 1] (float) or [0, 255]
    (uint8); the function detects the range automatically.

    Parameters
    ----------
    image : np.ndarray
        Image data to save.
        - Grayscale : shape (H, W)   – scalar pixel values.
        - RGB       : shape (H, W, 3).
        - RGBA      : shape (H, W, 4).
        Dtype may be float32 / float64 (values in [0, 1]) or uint8
        (values in [0, 255]).
    path : str
        Destination file path.  The extension determines the format:
        ``.png`` → lossless PNG; ``.jpg`` / ``.jpeg`` → lossy JPEG.
    cmap : str or None, optional
        Matplotlib colormap name used **only** for grayscale images.
        Defaults to ``'gray'`` when the image is 2-D and *cmap* is None.
        Pass ``None`` (default) for RGB images.
    quality : int, optional
        JPEG compression quality in [1, 95].  Ignored for PNG.
        Default is 95.

    Returns
    -------
    None

    Raises
    ------
    TypeError
        If *image* is not a NumPy ndarray, or *path* is not a str.
    ValueError
        If *path* is empty, the extension is unsupported, *quality* is
        outside [1, 95], or the image shape is invalid.
    OSError
        If the directory does not exist or the process lacks write
        permissions.

    Notes
    -----
    * Matplotlib's ``imsave`` is used under the hood.
    * For float arrays the values are expected in [0.0, 1.0].
      Values outside this range are clipped silently before saving.
    * JPEG does **not** support an alpha channel; RGBA images are
      composited onto a white background before JPEG export.

    Examples
    --------
    >>> save_image(img, "output/result.png")
    >>> save_image(gray, "output/gray.png", cmap="gray")
    >>> save_image(img, "output/result.jpg", quality=90)
    """
    _validate_path_string(path, "path")
    _validate_ndarray(image, "image")
    _validate_image_shape(image, "image")

    if not isinstance(quality, int) or not (1 <= quality <= 95):
        raise ValueError(
            f"'quality' must be an int in [1, 95], got {quality!r}."
        )

    ext = os.path.splitext(path)[1].lower()
    if ext not in (".png", ".jpg", ".jpeg"):
        raise ValueError(
            f"Unsupported file extension '{ext}'. "
            "Supported formats: .png, .jpg, .jpeg"
        )

    # Ensure parent directory exists
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        raise OSError(
            f"Destination directory does not exist: '{parent}'. "
            "Create it before saving."
        )

    # Work on a copy; normalise to float32 [0, 1]
    img = image.astype(np.float32)
    if img.max() > 1.0:                  # assume uint8-range [0, 255]
        img = img / 255.0
    img = np.clip(img, 0.0, 1.0)

    # JPEG does not support RGBA → composite onto white
    if ext in (".jpg", ".jpeg") and img.ndim == 3 and img.shape[2] == 4:
        rgb   = img[..., :3]
        alpha = img[..., 3:4]
        img   = rgb * alpha + (1.0 - alpha)   # white background

    # Choose colormap for grayscale
    effective_cmap = cmap
    if img.ndim == 2 and effective_cmap is None:
        effective_cmap = "gray"

    save_kwargs: dict = {}
    if ext in (".jpg", ".jpeg"):
        save_kwargs["pil_kwargs"] = {"quality": quality}

    plt.imsave(path, img, cmap=effective_cmap, **save_kwargs)


# ---------------------------------------------------------------------------
# 2.3  Color conversion
# ---------------------------------------------------------------------------

def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Convert an RGB or RGBA image to a 2-D grayscale array.

    Applies the ITU-R BT.601 luminance formula:

        Y = 0.2989 · R  +  0.5870 · G  +  0.1140 · B

    This matches the perceptual sensitivity of the human visual system
    (more weight on green, less on blue).

    Parameters
    ----------
    image : np.ndarray
        Input image of shape (H, W, 3) for RGB or (H, W, 4) for RGBA.
        Dtype may be float32 / float64 (range [0, 1]) or uint8 ([0, 255]).
        A 2-D grayscale image is returned **unchanged**.

    Returns
    -------
    np.ndarray
        Grayscale image of shape (H, W), same dtype as input.

    Raises
    ------
    TypeError
        If *image* is not a NumPy ndarray.
    ValueError
        If *image* has an incompatible shape (not 2-D or 3-D with C ∈ {1,3,4}).

    Notes
    -----
    * The alpha channel of RGBA images is **discarded** before conversion.
    * Single-channel (H, W, 1) arrays are squeezed to (H, W) directly,
      without reweighting.

    Examples
    --------
    >>> rgb  = read_image("photo.png")          # (H, W, 3)
    >>> gray = to_grayscale(rgb)                # (H, W)
    >>> gray.shape
    (480, 640)
    """
    _validate_ndarray(image, "image")
    _validate_image_shape(image, "image")

    if image.ndim == 2:
        return image                            # already grayscale

    if image.ndim == 3 and image.shape[2] == 1:
        return image[:, :, 0]                  # single-channel squeeze

    # Use only the first 3 channels (discard alpha if present)
    rgb = image[:, :, :3]

    # ITU-R BT.601 weights – fully vectorised, no Python loops
    weights = np.array([0.2989, 0.5870, 0.1140], dtype=np.float64)
    gray    = (rgb.astype(np.float64) * weights).sum(axis=2)

    return gray.astype(image.dtype)


def to_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert a 2-D grayscale image to a 3-channel RGB array by replication.

    Each output channel is identical to the grayscale input, producing a
    visually neutral (achromatic) RGB image.  This is useful when a
    downstream function requires a 3-channel input regardless of content.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image of shape (H, W) or single-channel (H, W, 1).
        Dtype may be float32 / float64 (range [0, 1]) or uint8 ([0, 255]).
        A 3-channel image is returned **unchanged**.

    Returns
    -------
    np.ndarray
        RGB image of shape (H, W, 3), same dtype as input.

    Raises
    ------
    TypeError
        If *image* is not a NumPy ndarray.
    ValueError
        If *image* has an incompatible shape.

    Examples
    --------
    >>> gray = read_image("gray.png", as_gray=True)  # (H, W)
    >>> rgb  = to_rgb(gray)                           # (H, W, 3)
    >>> rgb.shape
    (480, 640, 3)
    """
    _validate_ndarray(image, "image")
    _validate_image_shape(image, "image")

    if image.ndim == 3 and image.shape[2] == 3:
        return image                            # already RGB

    if image.ndim == 3 and image.shape[2] == 1:
        image = image[:, :, 0]                 # squeeze to (H, W)

    if image.ndim == 3 and image.shape[2] == 4:
        # RGBA → RGB (discard alpha, no conversion needed)
        return image[:, :, :3]

    # Stack the single channel three times along a new axis – no loops
    return np.stack([image, image, image], axis=2)


# ---------------------------------------------------------------------------
# Bonus utility: drop alpha channel
# ---------------------------------------------------------------------------

def drop_alpha(image: np.ndarray) -> np.ndarray:
    """
    Remove the alpha channel from an RGBA image, returning a plain RGB array.

    Parameters
    ----------
    image : np.ndarray
        Image of shape (H, W, 4).  Non-RGBA inputs are returned unchanged.

    Returns
    -------
    np.ndarray
        Image of shape (H, W, 3) if input was RGBA, otherwise the original
        array (no copy is made).

    Raises
    ------
    TypeError
        If *image* is not a NumPy ndarray.
    ValueError
        If the image shape is invalid.

    Examples
    --------
    >>> rgba = read_image("image_with_transparency.png")  # (H, W, 4)
    >>> rgb  = drop_alpha(rgba)                           # (H, W, 3)
    """
    _validate_ndarray(image, "image")
    _validate_image_shape(image, "image")

    if image.ndim == 3 and image.shape[2] == 4:
        return image[:, :, :3]

    return image                                # nothing to drop
