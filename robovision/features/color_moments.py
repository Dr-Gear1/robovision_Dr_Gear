"""
robovision/features/color_moments.py
======================================
Color Moments feature descriptor.

Computes the first three statistical moments of each color channel:
    - Mean       (1st moment) — average color
    - Standard deviation (2nd moment) — color spread
    - Skewness   (3rd moment) — color distribution asymmetry

For an RGB image: 3 channels × 3 moments = 9-D feature vector.

Reference: Stricker & Orengo, 1995 — "Similarity of Color Images".
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


# ── Color Moments ─────────────────────────────────────────────────────────

def extract_color_moments(
    image: np.ndarray,
    channels: str = "all",
    order: int = 3,
) -> np.ndarray:
    """
    Compute color moment features for each channel.

    The three moments are defined as follows for a channel with N pixels
    p_i ∈ [0, 1]:

        Mean       μ  = (1/N) Σ p_i
        Std dev    σ  = sqrt( (1/N) Σ (p_i - μ)² )
        Skewness   s  = cbrt( (1/N) Σ (p_i - μ)³ )

    Cube root (not square root) is used for skewness so the sign is
    preserved (skewness can be negative).

    Parameters
    ----------
    image : np.ndarray
        Input image, shape (H, W) for grayscale or (H, W, C) for color.
        dtype float [0, 1] or uint8 [0, 255] — normalised automatically.
    channels : {'all', 'rgb', 'gray'}, optional
        'all'  — use all available channels (default).
        'rgb'  — use only first 3 channels (drops alpha if RGBA).
        'gray' — convert to grayscale, compute single set of moments.
    order : {1, 2, 3}, optional
        Highest moment to compute.
        1 → mean only (1 value/channel).
        2 → mean + std (2 values/channel).
        3 → mean + std + skewness (3 values/channel).  Default 3.

    Returns
    -------
    np.ndarray
        1-D feature vector, dtype float32.
        Length = n_channels × order.
        For RGB with order=3: shape (9,).

    Raises
    ------
    TypeError
        If image is not ndarray.
    ValueError
        If order not in {1, 2, 3} or channels invalid.

    Notes
    -----
    * Very compact descriptor — only 9 values for RGB.
    * Sensitive to global illumination changes; combine with HOG for
      robust recognition.
    * Higher-order moments (order 4 = kurtosis) are not included here
      but can be added easily.

    Examples
    --------
    >>> feat = extract_color_moments(img)
    >>> feat.shape
    (9,)   # 3 channels × 3 moments
    >>> feat
    array([0.521, 0.184, -0.043, ...])  # mean, std, skew per channel
    """
    if order not in (1, 2, 3):
        raise ValueError(f"order must be 1, 2, or 3, got {order}.")
    if channels not in ("all", "rgb", "gray"):
        raise ValueError(f"channels must be 'all', 'rgb', or 'gray', got '{channels}'.")

    img = _validate(image)

    # Build channel list
    if channels == "gray" or img.ndim == 2:
        if img.ndim == 3:
            w   = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)
            gray = (img[:, :, :3] * w).sum(axis=2)
            channel_list = [gray.ravel()]
        else:
            channel_list = [img.ravel()]
    elif channels == "rgb":
        channel_list = [img[:, :, c].ravel() for c in range(min(3, img.shape[2]))]
    else:  # all
        if img.ndim == 2:
            channel_list = [img.ravel()]
        else:
            channel_list = [img[:, :, c].ravel() for c in range(img.shape[2])]

    features = []
    for pixels in channel_list:
        mu = pixels.mean()                           # 1st moment: mean

        if order >= 2:
            sigma = pixels.std()                     # 2nd moment: std deviation

        if order >= 3:
            diff = pixels - mu
            # Cube-root preserves sign of skewness
            skew_raw = (diff ** 3).mean()
            skew = np.cbrt(skew_raw)                 # 3rd moment: skewness

        if order == 1:
            features.extend([mu])
        elif order == 2:
            features.extend([mu, sigma])
        else:
            features.extend([mu, sigma, skew])

    return np.array(features, dtype=np.float32)


def extract_color_moments_hsv(image: np.ndarray) -> np.ndarray:
    """
    Compute color moments in HSV color space (9-D vector).

    HSV moments are more perceptually meaningful than RGB moments
    because Hue captures the actual color, Saturation captures
    the vividness, and Value captures the brightness.

    Parameters
    ----------
    image : np.ndarray
        RGB image, shape (H, W, 3) or (H, W, 4), float [0,1] or uint8.

    Returns
    -------
    np.ndarray
        9-D feature vector [H_mean, H_std, H_skew,
                            S_mean, S_std, S_skew,
                            V_mean, V_std, V_skew], dtype float32.

    Raises
    ------
    TypeError / ValueError
        On invalid input.

    Notes
    -----
    * For grayscale images, H and S will be 0, only V carries information.
    * Hue is circular — mean of hue can be misleading near 0°/360°.
      The mean here is a simple arithmetic mean, not a circular mean.

    Examples
    --------
    >>> feat = extract_color_moments_hsv(img)
    >>> feat.shape
    (9,)
    """
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be np.ndarray, got {type(image).__name__}.")
    img = image.astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0
    if img.ndim == 2:
        # Grayscale → treat as achromatic HSV
        V = img.ravel()
        H = np.zeros_like(V)
        S = np.zeros_like(V)
    elif img.ndim == 3:
        rgb = img[:, :, :3]
        R, G, B = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]

        # Value
        V = np.maximum(np.maximum(R, G), B)

        # Saturation
        min_c = np.minimum(np.minimum(R, G), B)
        delta = V - min_c
        S = np.where(V > 0, delta / (V + 1e-6), 0.0)

        # Hue (normalised to [0, 1])
        H = np.zeros_like(V)
        mask_r = (V == R) & (delta > 0)
        mask_g = (V == G) & (delta > 0)
        mask_b = (V == B) & (delta > 0)
        H[mask_r] = ((G[mask_r] - B[mask_r]) / (delta[mask_r] + 1e-6)) % 6
        H[mask_g] = (B[mask_g] - R[mask_g]) / (delta[mask_g] + 1e-6) + 2
        H[mask_b] = (R[mask_b] - G[mask_b]) / (delta[mask_b] + 1e-6) + 4
        H /= 6.0   # normalise to [0, 1]

        H, S, V = H.ravel(), S.ravel(), V.ravel()
    else:
        raise ValueError(f"image must be 2D or 3D, got shape {image.shape}.")

    features = []
    for channel in (H, S, V):
        mu      = channel.mean()
        sigma   = channel.std()
        diff    = channel - mu
        skew    = np.cbrt((diff ** 3).mean())
        features.extend([mu, sigma, skew])

    return np.array(features, dtype=np.float32)
