# RoboVision — API Reference

> **Version:** 1.0.0  
> **Author:** Omar Mustafa Mohammed  
> **Convention:** All functions accept and return `numpy.ndarray`.  
> Pixel values are `float32` in `[0.0, 1.0]` unless otherwise noted.

---

## Table of Contents

- [Module: `io.image_io`](#module-ioimage_io)
- [Module: `transforms.resize`](#module-transformsresize)
- [Module: `transforms.rotate`](#module-transformsrotate)
- [Module: `transforms.translate`](#module-transformstranslate)
- [Module: `transforms.flip`](#module-transformsflip)
- [Module: `transforms.pyramid`](#module-transformspyramid)
- [Module: `filters.filters`](#module-filtersfilters)
- [Module: `filters.thresholding`](#module-filtersthresholding)
- [Module: `filters.edge_detection`](#module-filtersedge_detection)
- [Module: `filters.histogram_ops`](#module-filtershistogram_ops)
- [Module: `features.hog`](#module-featureshog)
- [Module: `features.sift`](#module-featuressift)
- [Module: `features.color_histogram`](#module-featurescolor_histogram)
- [Module: `features.color_moments`](#module-featurescolor_moments)
- [Module: `features.spatial_pyramid`](#module-featuresspatial_pyramid)
- [Module: `utils.normalization`](#module-utilsnormalization)
- [Module: `utils.pixel_clipping`](#module-utilspixel_clipping)
- [Module: `utils.padding`](#module-utilspadding)
- [Module: `utils.convolution`](#module-utilsconvolution)
- [Module: `utils.drawing_primitives`](#module-utilsdrawing_primitives)
- [Module: `utils.text_placement`](#module-utilstext_placement)

---

## Module: `io.image_io`

> `robovision/io/image_io.py`  
> Image loading, saving, and colour space conversion.

---

### `read_image`

```python
read_image(path: str, as_gray: bool = False) -> np.ndarray
```

Load an image from disk into a NumPy array.

| Parameter | Type | Description |
|---|---|---|
| `path` | `str` | Absolute or relative path to the image file (`.png`, `.jpg`, `.jpeg`) |
| `as_gray` | `bool` | If `True`, return 2-D grayscale array. Default `False` |

**Returns:** `np.ndarray` — shape `(H,W,3)` for RGB, `(H,W)` for grayscale. Always `float32 [0,1]`.

**Raises:**
- `TypeError` — if `path` is not a string
- `FileNotFoundError` — if the file does not exist
- `OSError` — if the file cannot be decoded

```python
img  = read_image("photo.jpg")             # (H, W, 3) float32
gray = read_image("photo.jpg", as_gray=True)  # (H, W) float32
```

---

### `save_image`

```python
save_image(image: np.ndarray, path: str, cmap: str | None = None, quality: int = 95) -> None
```

Export a NumPy array to an image file on disk.

| Parameter | Type | Description |
|---|---|---|
| `image` | `np.ndarray` | Shape `(H,W)` or `(H,W,3)` or `(H,W,4)`. Float `[0,1]` or uint8 `[0,255]` |
| `path` | `str` | Destination path. Extension determines format: `.png` or `.jpg`/`.jpeg` |
| `cmap` | `str \| None` | Matplotlib colormap for grayscale images. Auto-set to `'gray'` if `None` |
| `quality` | `int` | JPEG compression quality in `[1, 95]`. Ignored for PNG. Default `95` |

**Returns:** `None`

**Raises:** `TypeError`, `ValueError` (unsupported extension, bad quality), `OSError` (directory missing)

> **Note:** RGBA images saved as JPEG are automatically composited onto a white background since JPEG has no alpha channel.

```python
save_image(img, "result.png")
save_image(img, "result.jpg", quality=90)
save_image(gray, "gray.png", cmap="gray")
```

---

### `to_grayscale`

```python
to_grayscale(image: np.ndarray) -> np.ndarray
```

Convert RGB/RGBA to 2-D grayscale using BT.601 luminance weights.

| Parameter | Type | Description |
|---|---|---|
| `image` | `np.ndarray` | Shape `(H,W,3)` or `(H,W,4)` or already `(H,W)` |

**Returns:** `np.ndarray` — shape `(H,W)`, same dtype as input.

> Formula: `Y = 0.2989·R + 0.5870·G + 0.1140·B`

```python
gray = to_grayscale(rgb_img)
```

---

### `to_rgb`

```python
to_rgb(image: np.ndarray) -> np.ndarray
```

Convert a grayscale `(H,W)` array to 3-channel `(H,W,3)` by channel replication.

```python
rgb = to_rgb(gray)    # all 3 channels are identical
```

---

### `drop_alpha`

```python
drop_alpha(image: np.ndarray) -> np.ndarray
```

Remove the alpha channel from an RGBA image. Returns non-RGBA inputs unchanged (no copy).

```python
rgb = drop_alpha(rgba_img)    # (H,W,4) → (H,W,3)
```

---

## Module: `transforms.resize`

> `robovision/transforms/resize.py`

---

### `resize`

```python
resize(image: np.ndarray, new_size: tuple, method: str = 'bilinear') -> np.ndarray
```

Resize an image using the specified interpolation method.

| Parameter | Type | Description |
|---|---|---|
| `image` | `np.ndarray` | Shape `(H,W)` or `(H,W,C)` |
| `new_size` | `tuple[int,int]` | Target `(height, width)`. Both values must be ≥ 1 |
| `method` | `str` | `'bilinear'` (default) or `'nearest'` |

**Returns:** `np.ndarray` — shape `(new_H, new_W)` or `(new_H, new_W, C)`. `float32` for bilinear; input dtype for nearest.

**Raises:** `ValueError` if method is not recognised or new_size values < 1.

```python
small  = resize(img, (128, 128))                      # bilinear
thumb  = resize(img, (64, 64), method='nearest')      # fast, blocky
large  = resize(img, (1024, 1024), method='bilinear') # smooth upscale
```

---

## Module: `transforms.rotate`

> `robovision/transforms/rotate.py`

---

### `rotate`

```python
rotate(image: np.ndarray, angle: float, method: str = 'bilinear', expand: bool = False) -> np.ndarray
```

Rotate an image about its centre (counter-clockwise).

| Parameter | Type | Description |
|---|---|---|
| `image` | `np.ndarray` | Input image |
| `angle` | `float` | Rotation angle in degrees (CCW positive) |
| `method` | `str` | `'bilinear'` (default) or `'nearest'` |
| `expand` | `bool` | If `True`, enlarge canvas to fit the full rotated image. Default `False` |

**Returns:** `np.ndarray` — same shape as input if `expand=False`; larger if `expand=True`. dtype `float32`.

> Pixels that map outside the original image are filled with `0` (black).

```python
rot45  = rotate(img, 45)
rot90  = rotate(img, 90, method='nearest')
full   = rotate(img, 45, expand=True)   # no corner cropping
```

---

## Module: `transforms.translate`

> `robovision/transforms/translate.py`

---

### `translate`

```python
translate(image: np.ndarray, tx: int, ty: int, fill: float = 0.0) -> np.ndarray
```

Shift an image by `(tx, ty)` pixels.

| Parameter | Type | Description |
|---|---|---|
| `tx` | `int` | Horizontal shift. Positive → right |
| `ty` | `int` | Vertical shift. Positive → down |
| `fill` | `float` | Value for newly exposed border pixels. Default `0.0` |

**Returns:** `np.ndarray` — same shape and dtype as input.

```python
right = translate(img, tx=50, ty=0)
diag  = translate(img, tx=30, ty=20, fill=0.5)
```

---

## Module: `transforms.flip`

> `robovision/transforms/flip.py`

---

### `flip`

```python
flip(image: np.ndarray, mode: str = 'horizontal') -> np.ndarray
```

| Parameter | Type | Description |
|---|---|---|
| `mode` | `str` | `'horizontal'` (mirror L-R), `'vertical'` (mirror T-B), `'both'` (180° rotation) |

**Returns:** `np.ndarray` — same shape and dtype. Always a copy, never a view.

```python
flipped_h = flip(img, mode='horizontal')
flipped_v = flip(img, mode='vertical')
rot180    = flip(img, mode='both')
```

---

## Module: `transforms.pyramid`

> `robovision/transforms/pyramid.py`

---

### `gaussian_pyramid`

```python
gaussian_pyramid(image: np.ndarray, levels: int = 4, sigma: float = 1.0) -> list[np.ndarray]
```

Build a Gaussian multi-scale pyramid.

**Returns:** `list` of length `levels`. Index `0` is original; each subsequent level is half the spatial resolution.

```python
pyr = gaussian_pyramid(img, levels=4)
# pyr[0].shape == (H, W, 3)
# pyr[1].shape == (H//2, W//2, 3)
```

---

### `laplacian_pyramid`

```python
laplacian_pyramid(image: np.ndarray, levels: int = 4, sigma: float = 1.0) -> list[np.ndarray]
```

Build a Laplacian pyramid (detail layers).  
Values may be negative. Collapse with `collapse_laplacian()` to reconstruct the original.

---

### `collapse_laplacian`

```python
collapse_laplacian(lap: list[np.ndarray]) -> np.ndarray
```

Reconstruct an image from a Laplacian pyramid. Returns `float32`.

---

## Module: `filters.filters`

> `robovision/filters/filters.py`  
> Core padding, convolution, and linear filters.

---

### `pad_image`

```python
pad_image(image: np.ndarray, pad_width: int, mode: str = 'reflect', constant_value: float = 0.0) -> np.ndarray
```

Pad an image on all four sides.

| `mode` | Behaviour |
|---|---|
| `'zero'` | Fill with 0 |
| `'reflect'` | Mirror, excluding edge pixel |
| `'replicate'` | Repeat edge pixel |
| `'constant'` | Fill with `constant_value` |
| `'circular'` | Wrap-around |

```python
padded = pad_image(img, pad_width=5)                              # reflect
padded = pad_image(img, pad_width=5, mode='constant', constant_value=0.5)
```

---

### `convolve2d`

```python
convolve2d(image: np.ndarray, kernel: np.ndarray, padding_mode: str = 'reflect') -> np.ndarray
```

True 2-D discrete convolution on a **grayscale** image (kernel is flipped 180°).

- Input: `(H, W)` only — use `apply_filter()` for RGB.
- Output: `(H, W)` float32, same spatial size ('same' convolution).

**Raises:** `ValueError` if image is 3-D.

```python
Gx = convolve2d(gray, np.array([[-1,0,1],[-2,0,2],[-1,0,1]], np.float32))
```

---

### `apply_filter`

```python
apply_filter(image: np.ndarray, kernel: np.ndarray, padding_mode: str = 'reflect') -> np.ndarray
```

Apply a kernel to grayscale or RGB (per-channel). Wrapper around `convolve2d`.

---

### `gaussian_kernel`

```python
gaussian_kernel(size: int, sigma: float) -> np.ndarray
```

Generate a normalised 2-D Gaussian kernel.

| Parameter | Type | Description |
|---|---|---|
| `size` | `int` | Side length (must be odd, ≥ 1) |
| `sigma` | `float` | Standard deviation (must be > 0) |

**Returns:** `np.ndarray` shape `(size, size)`, dtype `float32`, sums to `1.0`.

```python
k = gaussian_kernel(size=9, sigma=2.0)
assert abs(k.sum() - 1.0) < 1e-6
```

---

### `mean_filter`

```python
mean_filter(image: np.ndarray, kernel_size: int = 3, padding_mode: str = 'reflect') -> np.ndarray
```

Box blur — every output pixel is the mean of its `kernel_size × kernel_size` neighbourhood.

```python
blurred = mean_filter(img, kernel_size=7)
```

---

### `gaussian_filter`

```python
gaussian_filter(image: np.ndarray, size: int = 5, sigma: float = 1.0, padding_mode: str = 'reflect') -> np.ndarray
```

Gaussian smoothing via separable 1-D convolution (row pass then column pass).

```python
smooth = gaussian_filter(img, size=9, sigma=2.0)
```

---

### `median_filter`

```python
median_filter(image: np.ndarray, kernel_size: int = 3, padding_mode: str = 'reflect') -> np.ndarray
```

Non-linear median filter. Excellent for salt-and-pepper noise removal.

```python
clean = median_filter(noisy_img, kernel_size=5)
```

---

## Module: `filters.thresholding`

> `robovision/filters/thresholding.py`

---

### `threshold_global`

```python
threshold_global(image: np.ndarray, thresh: float, mode: str = 'binary') -> np.ndarray
```

Fixed global threshold. `thresh` must be in `[0, 1]`.

| `mode` | Rule |
|---|---|
| `'binary'` | `> thresh → 1.0`, else `0.0` |
| `'binary_inv'` | `> thresh → 0.0`, else `1.0` |
| `'trunc'` | `> thresh → thresh`, else unchanged |
| `'tozero'` | `> thresh → unchanged`, else `0.0` |
| `'tozero_inv'` | `> thresh → 0.0`, else unchanged |

```python
binary = threshold_global(gray, thresh=0.5)
inv    = threshold_global(gray, thresh=0.5, mode='binary_inv')
```

---

### `threshold_otsu`

```python
threshold_otsu(image: np.ndarray, n_bins: int = 256, return_thresh: bool = False)
    -> np.ndarray | tuple[np.ndarray, float]
```

Automatically find and apply the optimal threshold (maximises between-class variance).

```python
binary       = threshold_otsu(gray)
binary, t    = threshold_otsu(gray, return_thresh=True)
print(f"Otsu threshold: {t:.4f}")
```

---

### `threshold_adaptive`

```python
threshold_adaptive(image: np.ndarray, block_size: int = 11, C: float = 0.02,
                   method: str = 'mean', sigma: float = 2.0) -> np.ndarray
```

Local thresholding — each pixel compared against its neighbourhood statistic.

| Parameter | Description |
|---|---|
| `block_size` | Neighbourhood size (odd integer). Larger → smoother threshold map |
| `C` | Constant subtracted from local mean. Positive → fewer foreground pixels |
| `method` | `'mean'` or `'gaussian'` |

```python
local = threshold_adaptive(gray, block_size=21, C=0.02, method='gaussian')
```

---

## Module: `filters.edge_detection`

> `robovision/filters/edge_detection.py`

---

### `sobel_gradients`

```python
sobel_gradients(image: np.ndarray, padding_mode: str = 'reflect') -> dict[str, np.ndarray]
```

Compute Sobel gradient images. Converts to grayscale if input is colour.

**Returns:** dictionary with keys:

| Key | Shape | Description |
|---|---|---|
| `'Gx'` | `(H,W)` | Horizontal gradient |
| `'Gy'` | `(H,W)` | Vertical gradient |
| `'magnitude'` | `(H,W)` | `sqrt(Gx² + Gy²)` — not normalised |
| `'angle'` | `(H,W)` | `arctan2(Gy, Gx)` in degrees |

```python
result = sobel_gradients(gray)
mag    = result['magnitude'] / result['magnitude'].max()   # normalise for display
```

---

### `bit_plane_slice`

```python
bit_plane_slice(image: np.ndarray, plane: int) -> np.ndarray
```

Extract a single bit-plane from a grayscale image.

| Parameter | Type | Description |
|---|---|---|
| `plane` | `int` | Bit index `[0, 7]`. `0`=LSB, `7`=MSB |

**Returns:** `np.ndarray` shape `(H,W)`, dtype `uint8`, values in `{0, 1}`.

```python
msb    = bit_plane_slice(gray, plane=7)   # dominant structure
all_8  = bit_plane_all(gray)              # shape (8, H, W)
```

---

### `canny`

```python
canny(image: np.ndarray, low_thresh: float = 0.05, high_thresh: float = 0.15,
      gaussian_size: int = 5, gaussian_sigma: float = 1.0,
      padding_mode: str = 'reflect') -> np.ndarray
```

Full Canny edge detection pipeline (5 stages).

| Parameter | Description |
|---|---|
| `low_thresh` | Lower hysteresis threshold `[0,1]`. Pixels below → always suppressed |
| `high_thresh` | Upper threshold `[0,1]`. Must be > `low_thresh`. Pixels above → always kept |
| `gaussian_size` | Pre-smoothing kernel size |
| `gaussian_sigma` | Pre-smoothing sigma |

**Returns:** Binary edge map, shape `(H,W)`, dtype `float32`, values in `{0.0, 1.0}`.

```python
edges  = canny(gray)                                    # defaults
edges2 = canny(img, low_thresh=0.03, high_thresh=0.10)  # more sensitive
```

---

## Module: `filters.histogram_ops`

> `robovision/filters/histogram_ops.py`

---

### `compute_histogram`

```python
compute_histogram(image: np.ndarray, n_bins: int = 256, normalize: bool = False,
                  channel: int | None = None) -> tuple[np.ndarray, np.ndarray]
```

**Returns:** `(hist, bin_centers)` — both shape `(n_bins,)`, dtype `float32`.

```python
hist, bins = compute_histogram(gray, n_bins=256, normalize=True)
```

---

### `histogram_equalization`

```python
histogram_equalization(image: np.ndarray, n_bins: int = 256) -> np.ndarray
```

Enhance contrast by mapping pixel values through the CDF.  
Input must be grayscale or will be converted. Returns `float32 [0,1]`.

```python
enhanced = histogram_equalization(gray)
```

---

### `histogram_matching`

```python
histogram_matching(source: np.ndarray, reference: np.ndarray, n_bins: int = 256) -> np.ndarray
```

Transform `source` so its histogram matches `reference`. Applied per channel for colour images.

```python
matched = histogram_matching(source_img, reference_img)
```

---

### `gamma_correction`

```python
gamma_correction(image: np.ndarray, gamma: float) -> np.ndarray
```

Power-law intensity transformation: `output = input^γ`. Values clipped to `[0,1]` first.

```python
bright = gamma_correction(img, gamma=0.45)   # linearise sRGB
dark   = gamma_correction(img, gamma=2.2)    # encode sRGB
```

---

## Module: `features.hog`

> `robovision/features/hog.py`

---

### `extract_hog`

```python
extract_hog(image: np.ndarray, cell_size: int = 8, block_size: int = 2,
            n_bins: int = 9, signed: bool = False) -> np.ndarray
```

Compute the HOG (Histogram of Oriented Gradients) descriptor.

| Parameter | Description |
|---|---|
| `cell_size` | Pixels per cell side. Default `8` |
| `block_size` | Cells per block side. Default `2` (2×2 block) |
| `n_bins` | Orientation histogram bins. Default `9` |
| `signed` | `False` = unsigned `[0°,180°)`, `True` = signed `[0°,360°)` |

**Returns:** 1-D `float32` HOG vector. Length depends on image size.

> **Note:** Pass a fixed-size image (e.g. 128×128) for consistent vector length across a dataset.

```python
img_fixed = resize(img, (128, 128))
feat = extract_hog(to_grayscale(img_fixed))
```

---

## Module: `features.sift`

> `robovision/features/sift.py`

---

### `extract_sift`

```python
extract_sift(image: np.ndarray, n_octaves: int = 4, n_scales: int = 3,
             sigma0: float = 1.6, contrast_thresh: float = 0.03,
             max_keypoints: int | None = 500) -> tuple[list[Keypoint], np.ndarray]
```

Detect SIFT keypoints and compute 128-D descriptors.

**Returns:**
- `keypoints` — list of `Keypoint` objects with `.x`, `.y`, `.scale`, `.angle`, `.response`
- `descriptors` — `np.ndarray` shape `(N, 128)`, dtype `float32`

```python
kps, descs = extract_sift(gray, max_keypoints=200)
print(len(kps), descs.shape)    # e.g.: 187  (187, 128)
```

---

### `sift_feature_vector`

```python
sift_feature_vector(image: np.ndarray, max_keypoints: int = 50, **kwargs) -> np.ndarray
```

Fixed-length 128-D descriptor by averaging the top keypoint descriptors.  
Returns all-zeros if no keypoints found.

```python
vec = sift_feature_vector(gray)    # always shape (128,)
```

---

## Module: `features.color_histogram`

> `robovision/features/color_histogram.py`

---

### `extract_color_histogram`

```python
extract_color_histogram(image: np.ndarray, n_bins: int = 32,
                        normalize: bool = True, channels: str = 'all') -> np.ndarray
```

Per-channel concatenated colour histogram.

| `channels` | Output |
|---|---|
| `'all'` | All channels (default) |
| `'rgb'` | First 3 channels only |
| `'gray'` | Single grayscale histogram |

**Returns:** 1-D `float32` vector. Length = `n_channels × n_bins`.

```python
feat = extract_color_histogram(img, n_bins=32)   # shape (96,) for RGB
```

---

## Module: `features.color_moments`

> `robovision/features/color_moments.py`

---

### `extract_color_moments`

```python
extract_color_moments(image: np.ndarray, channels: str = 'all', order: int = 3) -> np.ndarray
```

Statistical moments per channel: mean, std, skewness.

| `order` | Output |
|---|---|
| `1` | Mean only — 1 value/channel |
| `2` | Mean + std — 2 values/channel |
| `3` | Mean + std + skewness — 3 values/channel (default) |

**Returns:** 1-D `float32` vector, shape `(n_channels × order,)`.

```python
feat = extract_color_moments(img)             # shape (9,)  for RGB
feat = extract_color_moments(img, order=1)    # shape (3,)  mean only
```

---

### `extract_color_moments_hsv`

```python
extract_color_moments_hsv(image: np.ndarray) -> np.ndarray
```

Color moments in HSV color space — 9-D vector `[H_μ, H_σ, H_s, S_μ, S_σ, S_s, V_μ, V_σ, V_s]`.

---

## Module: `features.spatial_pyramid`

> `robovision/features/spatial_pyramid.py`

---

### `extract_spatial_pyramid`

```python
extract_spatial_pyramid(image: np.ndarray, levels: int = 3, n_bins: int = 16,
                        descriptor: str = 'color', channels: str = 'rgb') -> np.ndarray
```

Spatial Pyramid Matching histogram.

| Parameter | Description |
|---|---|
| `levels` | Number of pyramid levels. Default `3` → grids 1×1, 2×2, 4×4 |
| `n_bins` | Histogram bins per channel |
| `descriptor` | `'color'` (default) or `'gray'` |

**Returns:** 1-D `float32` vector. For RGB, 16 bins, 3 levels: shape `(1008,)`.

```python
feat = extract_spatial_pyramid(img, levels=3, n_bins=16)
```

---

## Module: `utils.normalization`

> `robovision/utils/normalization.py`

---

### `normalize`

```python
normalize(image: np.ndarray, mode: str = 'minmax', **kwargs) -> np.ndarray
```

Unified normalization entry point.

| `mode` | Function called | Output dtype |
|---|---|---|
| `'minmax'` | `normalize_minmax()` | `float32` |
| `'zscore'` | `normalize_zscore()` | `float32` |
| `'scale_01'` | `normalize_scale(target='0-1')` | `float32` |
| `'scale_255'` | `normalize_scale(target='0-255')` | `uint8` |

```python
norm  = normalize(img, mode='minmax')
znorm = normalize(img, mode='zscore')
u8    = normalize(img, mode='scale_255')
```

---

## Module: `utils.pixel_clipping`

> `robovision/utils/pixel_clipping.py`

---

### Quick Reference

| Function | Signature | Use case |
|---|---|---|
| `clip` | `(img, low, high)` | Hard clamp to `[low, high]` |
| `clip_percentile` | `(img, low_pct=2, high_pct=98)` | Remove outlier intensities |
| `clip_sigma` | `(img, n_sigma=3.0)` | Statistical outlier removal |
| `clip_uint8` | `(img)` | Clamp to `[0,255]`, cast to `uint8` |

```python
safe  = clip(filtered_img, 0.0, 1.0)
clean = clip_percentile(img, low_pct=1, high_pct=99)
disp  = clip_uint8(img * 255)
```

---

## Module: `utils.padding`

> `robovision/utils/padding.py`

---

### `pad_image`

```python
pad_image(image: np.ndarray, pad_width: int, mode: str = 'reflect',
          constant_value: float = 0.0) -> np.ndarray
```

Five padding modes: `'zero'`, `'reflect'`, `'replicate'`, `'constant'`, `'circular'`.

Output size: `(H + 2·pad_width, W + 2·pad_width)`.

```python
p = pad_image(img, 10, mode='reflect')
p = pad_image(img, 10, mode='constant', constant_value=0.5)
```

---

### `unpad_image`

```python
unpad_image(padded: np.ndarray, pad_width: int) -> np.ndarray
```

Strip padding from all four sides. Returns a NumPy view (no copy).

---

## Module: `utils.convolution`

> `robovision/utils/convolution.py`

---

### `validate_kernel`

```python
validate_kernel(kernel: np.ndarray) -> None
```

Validate a convolution kernel. Raises `TypeError` or `ValueError` if invalid.  
Checks: must be ndarray, 2-D, non-empty, odd dimensions, numeric dtype.

---

### `convolve2d`

```python
convolve2d(image: np.ndarray, kernel: np.ndarray, padding_mode: str = 'reflect',
           constant_value: float = 0.0) -> np.ndarray
```

True 2-D convolution with kernel flip. Grayscale `(H,W)` images only.

---

### `filter2d`

```python
filter2d(image: np.ndarray, kernel: np.ndarray, padding_mode: str = 'reflect',
         constant_value: float = 0.0) -> np.ndarray
```

Cross-correlation (no kernel flip). Use for template matching.

---

### `spatial_filter`

```python
spatial_filter(image: np.ndarray, kernel: np.ndarray, padding_mode: str = 'reflect',
               constant_value: float = 0.0, rgb_strategy: str = 'per_channel') -> np.ndarray
```

Main spatial filtering entry point. Handles grayscale and RGB.

| `rgb_strategy` | Behaviour | Output shape |
|---|---|---|
| `'per_channel'` | Apply kernel to each channel independently | Same as input |
| `'luminance'` | Convert to grayscale first, then filter | `(H,W)` |

```python
blurred = spatial_filter(img, gaussian_kernel(9, 2.0))
edges   = spatial_filter(img, sobel_x, rgb_strategy='luminance')
```

---

## Module: `utils.drawing_primitives`

> `robovision/utils/drawing_primitives.py`  
> All functions draw **in-place** and also return the canvas.

**Color convention:**
- Grayscale canvas → scalar (e.g. `0.8`)
- RGB canvas → 3-tuple (e.g. `(1.0, 0.5, 0.0)`)
- Values match canvas dtype range: `[0,1]` for float, `[0,255]` for uint8

---

### `draw_point`

```python
draw_point(canvas, x: int, y: int, color, radius: int = 0) -> np.ndarray
```

| `radius` | Behaviour |
|---|---|
| `0` | Single pixel |
| `> 0` | Filled circle of that radius (vectorised mask) |

```python
draw_point(canvas, 100, 80, (1,0,0), radius=5)
```

---

### `draw_line`

```python
draw_line(canvas, x0, y0, x1, y1, color, thickness: int = 1) -> np.ndarray
```

Bresenham integer line. Thickness > 1 draws parallel offset lines.

```python
draw_line(canvas, 10, 10, 390, 290, (0,1,0), thickness=3)
```

---

### `draw_line_aa`

```python
draw_line_aa(canvas, x0, y0, x1, y1, color, thickness: int = 1) -> np.ndarray
```

Xiaolin Wu anti-aliased line. Sub-pixel accurate. Best on float canvases.

```python
draw_line_aa(canvas, 10.5, 10.0, 300.5, 200.0, (1,1,0))
```

---

### `draw_rectangle`

```python
draw_rectangle(canvas, x0, y0, x1, y1, color, filled: bool = False,
               thickness: int = 1) -> np.ndarray
```

| `filled` | Behaviour |
|---|---|
| `False` | Outline only (4 lines) |
| `True` | Filled rectangle (vectorised slice) |

```python
draw_rectangle(canvas, 50, 50, 200, 150, (0,0,1), filled=True)
draw_rectangle(canvas, 50, 50, 200, 150, (1,1,0), thickness=2)
```

---

### `draw_polygon`

```python
draw_polygon(canvas, points: list[tuple], color, filled: bool = False,
             thickness: int = 1) -> np.ndarray
```

`points` — list of `(x, y)` tuples. At least 2 required.  
Fill uses the scanline algorithm with even-odd rule.

```python
pts = [(50,50),(200,30),(300,120),(150,180),(30,120)]
draw_polygon(canvas, pts, (1,0.5,0), filled=True)
```

---

### `draw_ellipse`

```python
draw_ellipse(canvas, cx, cy, rx, ry, color, filled: bool = False,
             thickness: int = 1) -> np.ndarray
```

| Parameter | Description |
|---|---|
| `cx, cy` | Centre (column, row) |
| `rx` | Horizontal semi-axis (must be ≥ 1) |
| `ry` | Vertical semi-axis (must be ≥ 1) |

Set `rx == ry` for a circle.  
Filled ellipse uses a vectorised `np.ogrid` mask.

```python
draw_ellipse(canvas, 200, 150, 80, 50, (1,0,0), filled=True)
draw_ellipse(canvas, 200, 150, 50, 50, (0,1,0))   # circle
```

---

## Module: `utils.text_placement`

> `robovision/utils/text_placement.py`  
> Built-in 5×7 bitmap font. No external font libraries required.

---

### `draw_text`

```python
draw_text(canvas: np.ndarray, text: str, x: int, y: int, color,
          scale: int = 1, background_color=None) -> np.ndarray
```

Render a text string on the canvas.

| Parameter | Description |
|---|---|
| `x, y` | Top-left corner of the text bounding box (column, row) |
| `scale` | Integer upscale factor. `1` → 5×7 px, `2` → 10×14 px, `3` → 15×21 px |
| `background_color` | Optional fill behind text. Same format as `color` |

**Supported characters:** ASCII 32–126 (all printable characters)  
**Unsupported characters:** rendered as a small rectangle outline

```python
draw_text(canvas, "RoboVision 1.0", x=10, y=10, color=(1,1,1), scale=2)
draw_text(canvas, "Score: 97%", x=5, y=5, color=(0,1,0), scale=1,
          background_color=(0,0,0))
```

---

### `get_text_size`

```python
get_text_size(text: str, scale: int = 1) -> tuple[int, int]
```

Compute `(width, height)` in pixels before drawing.  
Useful for centring or right-aligning text.

```python
w, h = get_text_size("Hello", scale=2)   # (58, 14)
draw_text(canvas, "Hello", x=(W - w)//2, y=10, color=(1,1,1), scale=2)
```

---

## Common Error Reference

| Exception | When raised | Example trigger |
|---|---|---|
| `TypeError` | Wrong input type | `read_image(123)` |
| `FileNotFoundError` | Image path missing | `read_image("x.png")` if not exists |
| `ValueError` | Bad shape / invalid param | `convolve2d(rgb_img, kernel)` (3-D input) |
| `ValueError` | Even kernel size | `gaussian_kernel(4, 1.0)` |
| `ValueError` | Threshold out of range | `threshold_global(gray, thresh=1.5)` |
| `ValueError` | gamma ≤ 0 | `gamma_correction(img, gamma=0)` |
| `ValueError` | Ellipse semi-axis < 1 | `draw_ellipse(c, 100,100, 0, 20, ...)` |
| `ValueError` | Polygon < 2 points | `draw_polygon(c, [(10,10)], ...)` |

---

*End of API Reference — RoboVision v1.0.0*
