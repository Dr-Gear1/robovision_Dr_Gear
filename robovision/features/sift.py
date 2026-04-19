"""
robovision/features/sift.py
============================
Simplified SIFT-inspired keypoint detector and descriptor.

This is a from-scratch NumPy implementation that follows the main ideas
of Lowe's SIFT paper (2004):
    1. Scale-space extrema detection via DoG (Difference of Gaussians).
    2. Keypoint localisation and filtering.
    3. Orientation assignment.
    4. Descriptor computation (4×4 spatial grid, 8-bin histograms → 128-D).

Note: This is a teaching/project implementation — not the patent-encumbered
SIFT binary.  Results are similar but not identical to cv2.SIFT.

Only NumPy — no OpenCV, no scipy.
"""

from __future__ import annotations
import numpy as np


# ── Gaussian utilities ────────────────────────────────────────────────────

def _gaussian_kernel_2d(sigma: float, size: int | None = None) -> np.ndarray:
    """2-D Gaussian kernel normalised to sum = 1."""
    if size is None:
        size = int(2 * np.ceil(3 * sigma) + 1)
    if size % 2 == 0:
        size += 1
    c = size // 2
    y, x = np.mgrid[-c:c + 1, -c:c + 1]
    k = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return (k / k.sum()).astype(np.float32)


def _gaussian_kernel_1d(sigma: float) -> np.ndarray:
    """1-D Gaussian kernel normalised to sum = 1."""
    size = int(2 * np.ceil(3 * sigma) + 1)
    if size % 2 == 0:
        size += 1
    c  = size // 2
    ax = np.arange(size, dtype=np.float32) - c
    k  = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    return (k / k.sum()).astype(np.float32)


def _convolve1d(image: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    """
    Apply a 1-D kernel along one axis using vectorised slicing.
    Memory cost: O(H × W) instead of O(H × W × k²).
    """
    k   = len(kernel)
    pad = k // 2
    if axis == 0:
        padded = np.pad(image, ((pad, pad), (0, 0)), mode='reflect')
        H      = image.shape[0]
        out    = sum(kernel[i] * padded[i:i + H, :] for i in range(k))
    else:
        padded = np.pad(image, ((0, 0), (pad, pad)), mode='reflect')
        W      = image.shape[1]
        out    = sum(kernel[i] * padded[:, i:i + W] for i in range(k))
    return out.astype(np.float32)


def _blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur — two 1-D passes (rows then cols)."""
    k1d = _gaussian_kernel_1d(sigma)
    tmp = _convolve1d(image, k1d, axis=0)
    return _convolve1d(tmp,  k1d, axis=1)


# ── Validation ────────────────────────────────────────────────────────────

def _to_gray(image: np.ndarray) -> np.ndarray:
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be np.ndarray, got {type(image).__name__}.")
    if image.ndim == 3:
        w = np.array([0.2989, 0.5870, 0.1140], dtype=np.float64)
        image = (image[:, :, :3].astype(np.float64) * w).sum(axis=2)
    elif image.ndim != 2:
        raise ValueError(f"image must be 2D or 3D, got shape {image.shape}.")
    img = image.astype(np.float32)
    if img.max() > 1.0:
        img /= 255.0
    return img


# ── Keypoint dataclass (plain dict for NumPy-only policy) ────────────────

class Keypoint:
    """
    Lightweight SIFT keypoint container.

    Attributes
    ----------
    x, y     : float   — sub-pixel location (column, row)
    scale    : float   — detection scale (sigma)
    octave   : int     — pyramid octave
    response : float   — DoG response strength
    angle    : float   — dominant orientation in degrees [0, 360)
    """
    __slots__ = ('x', 'y', 'scale', 'octave', 'response', 'angle')

    def __init__(self, x, y, scale, octave, response, angle=0.0):
        self.x        = float(x)
        self.y        = float(y)
        self.scale    = float(scale)
        self.octave   = int(octave)
        self.response = float(response)
        self.angle    = float(angle)

    def __repr__(self):
        return (f"Keypoint(xy=({self.x:.1f},{self.y:.1f}), "
                f"scale={self.scale:.2f}, angle={self.angle:.1f}°, "
                f"resp={self.response:.4f})")


# ── Scale space construction ──────────────────────────────────────────────

def _build_scale_space(
    image: np.ndarray,
    n_octaves: int,
    n_scales: int,
    sigma0: float,
) -> tuple[list[list[np.ndarray]], list[list[np.ndarray]]]:
    """
    Build Gaussian scale-space and DoG pyramids.

    Returns
    -------
    gauss : list[list[ndarray]]   — gaussian[octave][scale]
    dogs  : list[list[ndarray]]   — dog[octave][scale]   (len = n_scales+2)
    """
    k = 2 ** (1.0 / n_scales)
    gauss_pyr = []
    dog_pyr   = []

    current = image.copy()
    for o in range(n_octaves):
        octave_gauss = []
        sigma = sigma0
        blurred = _blur(current, sigma)
        octave_gauss.append(blurred)

        for s in range(1, n_scales + 3):
            sigma_next = sigma0 * (k ** s)
            b = _blur(current, sigma_next)
            octave_gauss.append(b)

        gauss_pyr.append(octave_gauss)
        dog_pyr.append([
            octave_gauss[s + 1] - octave_gauss[s]
            for s in range(len(octave_gauss) - 1)
        ])

        # Downsample by 2 for next octave
        current = octave_gauss[n_scales][::2, ::2]

    return gauss_pyr, dog_pyr


# ── Extrema detection ─────────────────────────────────────────────────────

def _find_extrema(
    dog_pyr: list[list[np.ndarray]],
    contrast_thresh: float,
    border: int,
) -> list[Keypoint]:
    """Detect local min/max in 3×3×3 DoG neighbourhood."""
    keypoints = []
    sigma0 = 1.6
    k = 2 ** (1.0 / (len(dog_pyr[0]) - 2))

    for o, dogs in enumerate(dog_pyr):
        scale_factor = 2 ** o
        for s in range(1, len(dogs) - 1):
            prev, curr, next_ = dogs[s - 1], dogs[s], dogs[s + 1]
            H, W = curr.shape

            for r in range(border, H - border):
                for c in range(border, W - border):
                    val = curr[r, c]
                    if abs(val) < contrast_thresh:
                        continue
                    cube = np.array([
                        prev[r-1:r+2, c-1:c+2],
                        curr[r-1:r+2, c-1:c+2],
                        next_[r-1:r+2, c-1:c+2],
                    ])
                    if val == cube.max() or val == cube.min():
                        sigma = sigma0 * (k ** s) * scale_factor
                        kp = Keypoint(
                            x=c * scale_factor,
                            y=r * scale_factor,
                            scale=sigma,
                            octave=o,
                            response=float(val),
                        )
                        keypoints.append(kp)
    return keypoints


# ── Orientation assignment ────────────────────────────────────────────────

def _assign_orientations(
    keypoints: list[Keypoint],
    gauss_pyr: list[list[np.ndarray]],
    n_scales: int,
) -> list[Keypoint]:
    """Assign dominant orientation to each keypoint using gradient histogram."""
    k = 2 ** (1.0 / n_scales)
    sigma0 = 1.6
    oriented = []

    for kp in keypoints:
        o = kp.octave
        s = max(1, min(
            int(round(np.log2(kp.scale / sigma0) / np.log2(k))),
            len(gauss_pyr[o]) - 2
        ))
        img = gauss_pyr[o][s]
        H, W = img.shape
        radius = int(round(3 * kp.scale / (2 ** o)))
        r0 = int(round(kp.y / (2 ** o)))
        c0 = int(round(kp.x / (2 ** o)))

        hist = np.zeros(36)
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                r = r0 + dr
                c = c0 + dc
                if not (1 <= r < H - 1 and 1 <= c < W - 1):
                    continue
                gx = img[r, c + 1] - img[r, c - 1]
                gy = img[r + 1, c] - img[r - 1, c]
                mag = np.sqrt(gx ** 2 + gy ** 2)
                ori = np.rad2deg(np.arctan2(gy, gx)) % 360.0
                bin_idx = int(ori // 10) % 36
                hist[bin_idx] += mag

        max_val = hist.max()
        for b in range(36):
            if hist[b] >= 0.8 * max_val:
                new_kp = Keypoint(
                    x=kp.x, y=kp.y,
                    scale=kp.scale,
                    octave=kp.octave,
                    response=kp.response,
                    angle=b * 10.0,
                )
                oriented.append(new_kp)

    return oriented


# ── 128-D descriptor computation ──────────────────────────────────────────

def _compute_descriptors(
    keypoints: list[Keypoint],
    gauss_pyr: list[list[np.ndarray]],
    n_scales: int,
) -> np.ndarray:
    """
    Compute 128-D SIFT descriptor for each keypoint.
    4×4 spatial grid of 8-bin gradient histograms = 128 values.
    """
    if not keypoints:
        return np.zeros((0, 128), dtype=np.float32)

    k       = 2 ** (1.0 / n_scales)
    sigma0  = 1.6
    descs   = []

    for kp in keypoints:
        o   = kp.octave
        s   = max(1, min(
            int(round(np.log2(kp.scale / sigma0) / np.log2(k))),
            len(gauss_pyr[o]) - 2
        ))
        img = gauss_pyr[o][s]
        H, W = img.shape

        r0  = int(round(kp.y / (2 ** o)))
        c0  = int(round(kp.x / (2 ** o)))
        ang = np.deg2rad(kp.angle)
        cos_a, sin_a = np.cos(ang), np.sin(ang)

        window = 8   # half-window
        hist = np.zeros((4, 4, 8), dtype=np.float32)

        for dr in range(-window, window):
            for dc in range(-window, window):
                # Rotate relative coordinates
                rot_r = cos_a * dr + sin_a * dc
                rot_c = -sin_a * dr + cos_a * dc

                # Map to 4×4 grid
                grid_r = (rot_r + window) / (2 * window / 4)
                grid_c = (rot_c + window) / (2 * window / 4)
                gr = int(grid_r)
                gc = int(grid_c)
                if not (0 <= gr < 4 and 0 <= gc < 4):
                    continue

                r = r0 + dr
                c = c0 + dc
                if not (1 <= r < H - 1 and 1 <= c < W - 1):
                    continue

                gx  = img[r, c + 1] - img[r, c - 1]
                gy  = img[r + 1, c] - img[r - 1, c]
                mag = np.sqrt(gx ** 2 + gy ** 2)
                ori = (np.rad2deg(np.arctan2(gy, gx)) - kp.angle) % 360.0
                bin_idx = int(ori // 45) % 8
                hist[gr, gc, bin_idx] += mag

        desc = hist.ravel()
        # L2 normalise, clip, renormalise (Lowe 2004)
        norm = np.linalg.norm(desc) + 1e-6
        desc = desc / norm
        desc = np.clip(desc, 0, 0.2)
        desc = desc / (np.linalg.norm(desc) + 1e-6)
        descs.append(desc.astype(np.float32))

    return np.array(descs, dtype=np.float32)


# ── Public API ────────────────────────────────────────────────────────────

def extract_sift(
    image: np.ndarray,
    n_octaves: int = 4,
    n_scales: int = 3,
    sigma0: float = 1.6,
    contrast_thresh: float = 0.03,
    max_keypoints: int | None = 500,
) -> tuple[list[Keypoint], np.ndarray]:
    """
    Detect SIFT keypoints and compute 128-D descriptors.

    Algorithm summary
    -----------------
    1. Build Gaussian scale-space (n_octaves × (n_scales+3) images).
    2. Build DoG (Difference of Gaussians) pyramid.
    3. Detect extrema in 3×3×3 DoG neighbourhoods.
    4. Filter low-contrast candidates.
    5. Assign dominant orientation via gradient histogram.
    6. Compute 128-D descriptor (4×4 grid × 8 orientation bins).

    Parameters
    ----------
    image : np.ndarray
        Input image, shape (H, W) or (H, W, C), any dtype.
        Converted to float32 grayscale [0, 1] internally.
    n_octaves : int, optional
        Number of scale-space octaves.  Default 4.
    n_scales : int, optional
        Scales per octave.  Default 3.
    sigma0 : float, optional
        Base blur sigma.  Default 1.6 (Lowe's recommendation).
    contrast_thresh : float, optional
        Minimum |DoG| response to keep a keypoint.  Default 0.03.
    max_keypoints : int or None, optional
        Return only the top-N keypoints by response.  Default 500.

    Returns
    -------
    keypoints : list of Keypoint
        Detected and oriented keypoints.
    descriptors : np.ndarray, shape (N, 128), dtype float32
        One 128-D descriptor per keypoint.

    Raises
    ------
    TypeError
        If image is not ndarray.
    ValueError
        If image has wrong number of dimensions.

    Notes
    -----
    * This is a simplified implementation — loop-heavy for clarity.
      For a dataset with many images, consider batch processing.
    * Results are similar to but not identical to cv2.SIFT_create().

    Examples
    --------
    >>> kps, descs = extract_sift(img)
    >>> print(len(kps), descs.shape)
    312  (312, 128)
    """
    gray = _to_gray(image)

    gauss_pyr, dog_pyr = _build_scale_space(gray, n_octaves, n_scales, sigma0)

    border = 5
    keypoints = _find_extrema(dog_pyr, contrast_thresh, border)

    if max_keypoints and len(keypoints) > max_keypoints:
        keypoints.sort(key=lambda kp: abs(kp.response), reverse=True)
        keypoints = keypoints[:max_keypoints]

    keypoints   = _assign_orientations(keypoints, gauss_pyr, n_scales)
    descriptors = _compute_descriptors(keypoints, gauss_pyr, n_scales)

    return keypoints, descriptors


def sift_feature_vector(
    image: np.ndarray,
    max_keypoints: int = 50,
    **kwargs,
) -> np.ndarray:
    """
    Return a fixed-length feature vector from SIFT descriptors.

    Aggregates up to *max_keypoints* descriptors by averaging them into
    a single 128-D vector.  Useful when a fixed-size input is needed
    (e.g. for a classifier).

    Parameters
    ----------
    image : np.ndarray
        Input image.
    max_keypoints : int
        Number of top keypoints to average.  Default 50.
    **kwargs
        Forwarded to :func:`extract_sift`.

    Returns
    -------
    np.ndarray
        128-D mean descriptor, dtype float32.
        All-zeros if no keypoints are found.

    Examples
    --------
    >>> vec = sift_feature_vector(img)
    >>> vec.shape
    (128,)
    """
    kps, descs = extract_sift(image, max_keypoints=max_keypoints, **kwargs)
    if len(descs) == 0:
        return np.zeros(128, dtype=np.float32)
    return descs.mean(axis=0).astype(np.float32)
