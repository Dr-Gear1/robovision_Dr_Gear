# RoboVision — Math & Algorithms Notes

> **Purpose:** Concise mathematical derivations and pseudocode for every algorithm implemented in RoboVision.  
> Each section maps directly to a source module so you can cross-reference code and theory side-by-side.

---

## Table of Contents

1. [Image Fundamentals](#1-image-fundamentals)
2. [Normalization](#2-normalization)
3. [Pixel Clipping](#3-pixel-clipping)
4. [Padding](#4-padding)
5. [2D Convolution](#5-2d-convolution)
6. [Spatial Filtering](#6-spatial-filtering)
7. [Image Filters](#7-image-filters)
8. [Thresholding](#8-thresholding)
9. [Edge Detection](#9-edge-detection)
10. [Histogram Operations](#10-histogram-operations)
11. [Geometric Transforms](#11-geometric-transforms)
12. [Image Pyramids](#12-image-pyramids)
13. [Feature Extractors](#13-feature-extractors)
14. [Drawing Primitives](#14-drawing-primitives)

---

## 1. Image Fundamentals

**Module:** `robovision/io/image_io.py`

### 1.1 Image Representation

An image is a 2-D or 3-D NumPy array:

```
Grayscale : I  ∈ ℝ^{H × W}         one intensity per pixel
RGB       : I  ∈ ℝ^{H × W × 3}     R, G, B channels
RGBA      : I  ∈ ℝ^{H × W × 4}     R, G, B, Alpha
```

All values are stored as **float32 in [0, 1]** after loading.  
Uint8 images (JPEG) are normalised on read: `pixel_float = pixel_uint8 / 255.0`

---

### 1.2 Grayscale Conversion — BT.601 Luminance

Converting RGB to grayscale uses perceptually weighted channel combination.  
The weights reflect human eye sensitivity (more sensitive to green, less to blue):

```
Y = 0.2989 · R  +  0.5870 · G  +  0.1140 · B
```

**Why these weights?**  
Derived from the ITU-R BT.601 standard. The human eye has ~64% sensitivity to green, ~21% to red, ~5% to blue. Simple averaging `(R+G+B)/3` looks washed out by comparison.

**Vectorised implementation:**
```python
weights = [0.2989, 0.5870, 0.1140]
Y = (RGB[:,:,:3] * weights).sum(axis=2)
```

---

## 2. Normalization

**Module:** `robovision/utils/normalization.py`

### 2.1 Min-Max Normalization

Maps the full pixel range to a target interval `[a, b]`:

```
           pixel - min(I)
pixel' = ─────────────────── · (b - a) + a
           max(I) - min(I)
```

- Special case: if `max(I) = min(I)` (constant image), set all pixels to `a`.
- Output is guaranteed to lie in `[a, b]`.

**Common targets:**
- `[0, 1]`   → standard float range for processing
- `[0, 255]` → display-ready uint8 range

---

### 2.2 Z-score Normalization

Standardises the image to zero mean and unit variance:

```
         pixel - μ
pixel' = ─────────
           σ + ε
```

where:
- `μ = mean of all pixels`
- `σ = std deviation of all pixels`
- `ε = 1e-8` (prevents division by zero for constant images)

**Output range:** unbounded — can be negative or > 1.  
**When to use:** before training neural networks; zero-centred inputs improve gradient flow.

---

### 2.3 Scale Normalization

Simple dtype-aware range conversion:

| Input detected | Target `'0-1'` | Target `'0-255'` |
|---|---|---|
| float, max ≤ 1.0 | keep as-is, clip | multiply by 255, cast uint8 |
| uint8, max > 1.0 | divide by 255.0 | keep as-is, clip |

---

## 3. Pixel Clipping

**Module:** `robovision/utils/pixel_clipping.py`

### 3.1 Hard Clip

```
pixel' = max(low, min(high, pixel))
```

Implemented as `np.clip(image, low, high)` — fully vectorised, O(H·W).

---

### 3.2 Percentile Clip

Computes the `p`-th and `(100-p)`-th percentile of all pixel values, then hard-clips:

```
lo = percentile(I, p)
hi = percentile(I, 100-p)
pixel' = clip(pixel, lo, hi)
```

**Use case:** removes extreme outlier intensities (e.g., dead pixels in thermal cameras) without knowing the exact data range.

---

### 3.3 Sigma Clip

Clips at ±N standard deviations from the mean:

```
lo = μ - N·σ
hi = μ + N·σ
pixel' = clip(pixel, lo, hi)
```

Standard value: `N = 3.0` (keeps 99.7% of a Gaussian distribution).

---

## 4. Padding

**Module:** `robovision/utils/padding.py`

Padding adds `p` pixels on each side. It is required before convolution to produce a **same-size output** (set `p = kernel_size // 2`).

| Mode | Formula (1-D example, p=2, signal=[a,b,c,d,e]) | Notes |
|---|---|---|
| `zero` | `0 0 \| a b c d e \| 0 0` | Simple; causes dark edges after filtering |
| `reflect` | `c b \| a b c d e \| d c` | No edge artefact; best default |
| `replicate` | `a a \| a b c d e \| e e` | Good near object boundaries |
| `constant` | `v v \| a b c d e \| v v` | User-specified fill value `v` |
| `circular` | `d e \| a b c d e \| a b` | Periodic; for signals with wrap-around |

---

## 5. 2D Convolution

**Module:** `robovision/utils/convolution.py`  
**Also:** `robovision/filters/filters.py`

### 5.1 Discrete 2D Convolution

True convolution of image `I` with kernel `K` of size `(2m+1) × (2n+1)`:

```
(I * K)[r, c] = Σ_{i=-m}^{m} Σ_{j=-n}^{n}  I[r-i, c-j] · K[i, j]
```

Note the **kernel flip**: indices `(r-i, c-j)` — equivalent to rotating K by 180° and sliding.  
For symmetric kernels (Gaussian, mean) flipping has no effect.

### 5.2 Cross-Correlation (no flip)

```
(I ⊛ K)[r, c] = Σ_{i} Σ_{j}  I[r+i, c+j] · K[i, j]
```

Used in template matching. `filter2d()` implements this.

### 5.3 Kernel Validation Rules

Before any convolution, the kernel must satisfy:
1. Type: `numpy.ndarray`
2. Shape: exactly 2-D
3. Non-empty: `kernel.size > 0`
4. Odd dimensions: `kH % 2 == 1` and `kW % 2 == 1`
5. Numeric dtype: `int` or `float`

### 5.4 Vectorised Implementation (stride tricks)

Instead of looping over each output pixel, all windows are extracted simultaneously:

```python
shape   = (H, W, kH, kW)
strides = (s0, s1, s0, s1)          # s0, s1 = padded array strides
windows = as_strided(padded, shape, strides)   # 4D view, no copy
output  = (windows * K_flipped).sum(axis=(2, 3))
```

Memory cost: `O(H · W · kH · kW)` — manageable for k ≤ 15 on typical images.

---

## 6. Spatial Filtering

**Module:** `robovision/utils/convolution.py` → `spatial_filter()`

### RGB Strategies

**Per-channel (default):**  
Apply the same kernel independently to R, G, B:
```
out[:,:,0] = convolve2d(I[:,:,0], K)
out[:,:,1] = convolve2d(I[:,:,1], K)
out[:,:,2] = convolve2d(I[:,:,2], K)
```
Correct for all linear filters. Output shape same as input.

**Luminance:**  
Convert to grayscale first, then filter:
```
gray = 0.2989·R + 0.5870·G + 0.1140·B
out  = convolve2d(gray, K)
```
Output is 2-D. Used when only edge intensity matters (e.g., Canny on colour images).

---

## 7. Image Filters

**Module:** `robovision/filters/filters.py`

### 7.1 Mean (Box) Filter

Kernel: all values equal `1/k²`

```
         1
K[i,j] = ────     for all i,j ∈ [0, k-1]
          k²
```

**Effect:** blurs by averaging every pixel with its neighbourhood.  
**Drawback:** does not preserve edges; ringing artefacts for large k.

---

### 7.2 Gaussian Filter

#### Kernel Generation

2-D Gaussian kernel (separable — outer product of two 1-D Gaussians):

```
             -(x² + y²)
G(x,y) = exp──────────────
               2σ²
```

Then normalise: `K = G / sum(G)` so the filter preserves image brightness.

**Rule of thumb for size:** `size = 2 · ceil(3σ) + 1`  
(captures ±3σ of the Gaussian, which contains 99.7% of the area)

#### Separable Filtering (performance)

Because `G(x,y) = G(x) · G(y)`, a 2-D Gaussian can be computed as two 1-D passes:

```
Step 1: blur rows    → tmp  = I ⊛ g(x)      O(H · W · k)
Step 2: blur columns → out  = tmp ⊛ g(y)    O(H · W · k)
```

Total: `O(H·W·k)` instead of `O(H·W·k²)` — a factor of `k` faster.

---

### 7.3 Median Filter

#### Why loops are justified

The median is a **non-linear** operation — it cannot be written as a dot product of a kernel with a pixel window. There is no fully vectorised equivalent.

**Implementation strategy (minimised looping):**

```python
# Extract all windows at once using stride tricks
windows = as_strided(padded, shape=(H,W,k,k), strides=(...))  # no copy
output  = np.median(windows, axis=(2,3))                        # one call
```

This avoids per-pixel Python loops — only one NumPy `median` call over the 4-D window array.

**Complexity:** `O(H · W · k² · log(k²))` — the sorting cost per window.

**Strengths:** excellent for salt-and-pepper noise; preserves edges far better than mean/Gaussian.

---

## 8. Thresholding

**Module:** `robovision/filters/thresholding.py`

### 8.1 Global Thresholding

Simple pixel-wise comparison:

```
          ┌ 1   if I[r,c] > t
out[r,c] = ┤
          └ 0   otherwise
```

Five modes available: `binary`, `binary_inv`, `trunc`, `tozero`, `tozero_inv`.

---

### 8.2 Otsu's Method

**Goal:** Find the threshold `t*` that maximises the **between-class variance** of the two pixel groups (background / foreground).

**Definitions:**
```
ω₀(t) = P(pixel ≤ t)               ← background weight
ω₁(t) = 1 - ω₀(t)                 ← foreground weight
μ₀(t) = E[pixel | pixel ≤ t]      ← background mean
μ₁(t) = E[pixel | pixel > t]      ← foreground mean
```

**Between-class variance:**
```
σ²_B(t) = ω₀(t) · ω₁(t) · (μ₀(t) - μ₁(t))²
```

**Optimal threshold:**
```
t* = argmax_t  σ²_B(t)
```

**Vectorised over all thresholds using cumulative sums:**
```python
w0  = cumsum(prob)                          # background weights, all t at once
mu0 = cumsum(prob * bin_centers) / (w0+ε)  # background means
mu1 = (total_mean - cumsum(prob*bins)) / (w1+ε)
sigma_b = w0 * w1 * (mu0 - mu1)**2
t_opt   = bin_centers[argmax(sigma_b)]
```

No loop over thresholds — all computed in one pass.

---

### 8.3 Adaptive Thresholding

For each pixel, the threshold is computed from its local neighbourhood:

```
thresh[r,c] = local_stat(neighbourhood(r,c)) - C
out[r,c]    = 1  if  I[r,c] > thresh[r,c]  else  0
```

Two local statistics:

| Method | Formula |
|---|---|
| `mean` | `thresh = mean of block - C` |
| `gaussian` | `thresh = Gaussian-weighted mean of block - C` |

`C` is a small positive constant (default 0.02) that fine-tunes sensitivity.  
Implemented via convolution — fully vectorised, no per-pixel loops.

---

## 9. Edge Detection

**Module:** `robovision/filters/edge_detection.py`

### 9.1 Sobel Gradients

3×3 Sobel kernels approximate the image gradient:

```
        [-1  0  1]            [-1 -2 -1]
Kx =    [-2  0  2]    Ky =    [ 0  0  0]
        [-1  0  1]            [ 1  2  1]
```

Applied via convolution:
```
Gx = I * Kx          (horizontal gradient)
Gy = I * Ky          (vertical gradient)

magnitude = sqrt(Gx² + Gy²)
angle     = arctan2(Gy, Gx)   [degrees, range: -180° to 180°]
```

The Sobel kernel combines Gaussian smoothing (reduces noise) with differentiation in one step.

---

### 9.2 Bit-Plane Slicing

Each pixel value `v ∈ [0, 255]` has 8 binary bits. The k-th bit-plane:

```
plane_k[r,c] = (v[r,c] >> k) & 1
```

| Plane | Significance |
|---|---|
| 7 (MSB) | Dominant structure — major edges |
| 6, 5, 4 | Most visual information |
| 3, 2, 1 | Fine detail, texture |
| 0 (LSB) | Noise, quantisation effects |

Fully vectorised: `(img_uint8 >> plane) & 1` — no loops.

---

### 9.3 Canny Edge Detector

**Five-stage pipeline:**

```
Stage 1 — Gaussian Smoothing
    Reduce noise before gradient computation.
    I_smooth = I * G(σ)

Stage 2 — Sobel Gradients
    Compute magnitude M and angle θ at every pixel.

Stage 3 — Non-Maximum Suppression (NMS)
    Thin edges to 1-pixel width.
    For each pixel: compare M with its two neighbours
    along the gradient direction θ.
    Keep pixel if M ≥ both neighbours; otherwise suppress to 0.

    Four quantised directions:
      0°   → compare East / West
     45°   → compare NE / SW
     90°   → compare North / South
    135°   → compare NW / SE

Stage 4 — Double Thresholding
    Classify pixels into three groups:
    - Strong : M ≥ high_thresh    → definite edge
    - Weak   : low ≤ M < high     → candidate edge
    - None   : M < low_thresh     → suppressed

Stage 5 — Hysteresis Edge Tracking
    Promote weak pixels to edges if connected (8-neighbour)
    to at least one strong pixel.
    Discard isolated weak pixels.
    Repeat until no more promotions.
```

**Why hysteresis?** A single threshold causes either missed edges (too high) or noisy blobs (too low). The dual threshold with connectivity tracking finds continuous edges robustly.

**Recommended ratio:** `high / low ≈ 2–3` (Canny's original recommendation).

---

## 10. Histogram Operations

**Module:** `robovision/filters/histogram_ops.py`

### 10.1 Histogram

Count of pixels per intensity bin:

```
hist[k] = number of pixels with value in [k/N, (k+1)/N)
```

Normalised histogram (probability):
```
p[k] = hist[k] / total_pixels
```

---

### 10.2 Histogram Equalization

**Goal:** redistribute intensities so the output histogram is approximately flat.

**Method — CDF mapping:**

```
CDF(k) = Σ_{i=0}^{k} p[i]          ← cumulative distribution function

output[r,c] = CDF( input[r,c] )
```

The CDF is a monotone mapping in [0, 1] — it stretches dense regions and compresses sparse ones.

**Proof of uniformity:** If `p` is uniform, then `CDF` is linear, and the mapping `CDF(input)` is the identity. For a non-uniform `p`, the CDF remapping pulls pixels from high-density bins into the full range.

---

### 10.3 Histogram Matching (Specification)

**Goal:** transform `source` image so its histogram matches `reference`.

**Method — Double CDF:**

```
T(u) = CDF_ref⁻¹( CDF_src(u) )

1. Compute CDF_src from source image
2. Compute CDF_ref from reference image
3. For each pixel u in source:
       find v = CDF_ref⁻¹(CDF_src(u))
       i.e. find v such that CDF_ref(v) ≈ CDF_src(u)
4. output[r,c] = T( source[r,c] )
```

Implemented via `np.interp` — fully vectorised, no pixel-by-pixel loop.

---

### 10.4 Gamma Correction

Power-law intensity transformation:

```
output = input^γ        (values in [0, 1])
```

| γ | Effect |
|---|---|
| < 1 | Brightens image (expands darks) |
| = 1 | Identity — no change |
| > 1 | Darkens image (compresses darks) |

**Common values:**
- `γ = 2.2` — sRGB display encoding
- `γ = 1/2.2 ≈ 0.45` — linearise sRGB for computation
- `γ = 0.5` — strong brightening

Single `np.power(img, gamma)` call — fully vectorised.

---

## 11. Geometric Transforms

**Module:** `robovision/transforms/`

### 11.1 Resizing

#### Nearest-Neighbour Interpolation

Map each output pixel to the closest input pixel:

```
r_src = round( r_out · H_in / H_out )
c_src = round( c_out · W_in / W_out )
output[r_out, c_out] = input[r_src, c_src]
```

Fast; produces blocky/pixelated results when upscaling.

#### Bilinear Interpolation

Map each output pixel to a continuous input coordinate, then blend 4 neighbours:

```
r_src = r_out · (H_in - 1) / (H_out - 1)
c_src = c_out · (W_in - 1) / (W_out - 1)

r0, r1 = floor(r_src), ceil(r_src)
c0, c1 = floor(c_src), ceil(c_src)
dr = r_src - r0
dc = c_src - c0

output = (1-dr)(1-dc)·I[r0,c0] + (1-dr)·dc·I[r0,c1]
       +    dr ·(1-dc)·I[r1,c0] +    dr ·dc·I[r1,c1]
```

Smooth results; slightly blurred at large upscale ratios.

---

### 11.2 Rotation

Uses **inverse mapping** to avoid holes in the output.

For output pixel `(r', c')`, find where it came from in the input:

```
cx = (W-1)/2,   cy = (H-1)/2        ← image centre

Δr = r' - cy,   Δc = c' - cx

r_src =  cos(θ)·Δr + sin(θ)·Δc  + cy
c_src = -sin(θ)·Δr + cos(θ)·Δc  + cx
```

Then sample `input[r_src, c_src]` with bilinear interpolation.  
Pixels that map outside the input are set to 0 (black).

**Why inverse mapping?** Forward mapping (push pixels) leaves gaps; inverse mapping (pull pixels) fills every output pixel.

---

### 11.3 Translation

Shift the image by `(tx, ty)` pixels. Implemented as slice copy:

```
output = zeros_like(input)
output[ty:, tx:] = input[:H-ty, :W-tx]     (for positive tx, ty)
```

Border pixels exposed by the shift are filled with a constant (default 0).  
Fully vectorised — single NumPy slice assignment.

---

## 12. Image Pyramids

**Module:** `robovision/transforms/pyramid.py`

### 12.1 Gaussian Pyramid

Multi-scale representation: each level is blurred and downsampled by 2×.

```
level[0] = original image
level[i] = downsample( gaussian_blur( level[i-1] ) )
         = level[i-1]_blurred[::2, ::2]
```

Blurring before downsampling is essential to avoid aliasing (Nyquist criterion — remove frequencies above the new Nyquist limit before subsampling).

**Why blur first?**  
Without blurring, downsampling by 2 aliases high-frequency content into lower frequencies, producing Moiré patterns.

---

### 12.2 Laplacian Pyramid

Encodes the **detail lost** at each Gaussian level:

```
lap[i]  = gauss[i] - upsample(gauss[i+1])
lap[L]  = gauss[L]    ← coarsest residual
```

**Reconstruction (collapse):** exactly recovers the original:

```
for i from L-1 down to 0:
    gauss[i] = lap[i] + upsample(gauss[i+1])
```

Used in: image blending, compression, multi-scale analysis.

---

## 13. Feature Extractors

**Module:** `robovision/features/`

### 13.1 HOG — Histogram of Oriented Gradients

> Reference: Dalal & Triggs, CVPR 2005

**Pipeline:**

```
Step 1: Convert to grayscale.

Step 2: Compute gradients at each pixel.
    Gx = I ⊛ [-1, 0, 1]
    Gy = I ⊛ [-1, 0, 1]ᵀ
    magnitude  = sqrt(Gx² + Gy²)
    orientation = atan2(Gy, Gx) mod 180°     (unsigned)

Step 3: Divide image into cells of size (c × c) pixels.
    For each cell: build a 9-bin orientation histogram.
    Each pixel votes for its bin proportional to its magnitude.
    Soft bilinear voting splits the vote between the two nearest bins.

Step 4: Group cells into (b × b) blocks.
    Concatenate the b² cell histograms → block descriptor.
    L2-Hys normalisation:
        v = v / ||v||          (L2 normalise)
        v = clip(v, 0, 0.2)   (saturate large values)
        v = v / ||v||          (renormalise)

Step 5: Concatenate all block descriptors → final HOG vector.
```

**Descriptor length** (for 128×128 image, cell=8, block=2, bins=9):
```
n_blocks_x = (128/8 - 2 + 1) = 14
n_blocks_y = (128/8 - 2 + 1) = 14
length = 14 × 14 × 2 × 2 × 9 = 7056
```

---

### 13.2 SIFT — Scale-Invariant Feature Transform

> Reference: Lowe, IJCV 2004

**Pipeline:**

```
Step 1: Build Gaussian scale-space.
    k = 2^(1/s)   (s = scales per octave, typically 3)
    For each octave o, scale s:
        gauss[o][s] = blur(image_at_octave, σ₀ · k^s)

Step 2: Build DoG (Difference of Gaussians) pyramid.
    DoG[o][s] = gauss[o][s+1] - gauss[o][s]

Step 3: Detect extrema.
    A pixel is a keypoint if it is a local max or min
    in its 3×3×3 neighbourhood (across scale and space).
    Filter by: |DoG| > contrast_threshold

Step 4: Assign dominant orientation.
    Compute gradient histogram (36 bins, 10° each)
    in a patch around the keypoint.
    Assign orientation of the dominant bin.

Step 5: Compute 128-D descriptor.
    Rotate patch by keypoint orientation (for rotation invariance).
    Divide into 4×4 sub-regions.
    Compute 8-bin gradient histogram per sub-region.
    Concatenate → 4 × 4 × 8 = 128-D vector.
    L2-normalise → clip at 0.2 → renormalise.
```

**Fixed-length vector:** average all descriptors → 128-D `sift_feature_vector()`.

---

### 13.3 Color Histogram

Per-channel pixel value histogram:

```
For each channel c in {R, G, B}:
    hist_c[k] = count(pixels in bin k)  / total_pixels

feature = [hist_R | hist_G | hist_B]    ← concatenate
```

With 32 bins per channel: **96-D vector** for RGB images.

---

### 13.4 Color Moments

First three statistical moments per channel — compact 9-D descriptor:

```
Mean       μ_c = (1/N) Σ p_i
Std Dev    σ_c = sqrt( (1/N) Σ (p_i - μ_c)² )
Skewness   s_c = cbrt( (1/N) Σ (p_i - μ_c)³ )    ← cube root preserves sign

feature = [μ_R, σ_R, s_R, μ_G, σ_G, s_G, μ_B, σ_B, s_B]
```

> Reference: Stricker & Orengo, 1995

**Why skewness (3rd moment)?** Describes asymmetry of the intensity distribution — a left-skewed image has more dark pixels; right-skewed has more bright pixels.

---

### 13.5 Spatial Pyramid Matching (SPM)

> Reference: Lazebnik, Schmid & Ponce, CVPR 2006

Captures the **spatial layout** of colours/gradients by computing histograms at multiple scales:

```
Level 0:  1×1  grid   →   1 cell
Level 1:  2×2  grid   →   4 cells
Level 2:  4×4  grid   →  16 cells
```

Each cell gets a colour histogram. The cells are weighted:

```
w[0]  = 1 / 2^L                 (coarsest level)
w[l]  = 1 / 2^(L-l+1)          (l ≥ 1)
```

**Final vector:** weighted concatenation of all cell histograms.

**Example size** (L=3, 16 bins, RGB):
```
Level 0:   1 × 3 × 16 =    48
Level 1:   4 × 3 × 16 =   192
Level 2:  16 × 3 × 16 =   768
Total                  = 1,008 dimensions
```

---

## 14. Drawing Primitives

**Module:** `robovision/utils/drawing_primitives.py`

### 14.1 Bresenham Line Algorithm

Draws a line using only integer arithmetic.

```
dx = |x1 - x0|,   dy = -|y1 - y0|
sx = sign(x1 - x0),   sy = sign(y1 - y0)
err = dx + dy

while True:
    plot(x0, y0)
    if x0 == x1 and y0 == y1: break
    e2 = 2 * err
    if e2 >= dy:  err += dy;  x0 += sx
    if e2 <= dx:  err += dx;  y0 += sy
```

The error term `err` tracks how far the actual line position deviates from the ideal path.

---

### 14.2 Wu Anti-Aliased Line

Xiaolin Wu's algorithm blends two adjacent pixels per step, weighted by sub-pixel distance:

```
For each x along the dominant axis:
    y_exact = y0 + gradient · (x - x0)
    y0_int  = floor(y_exact)
    frac    = y_exact - y0_int         ← fractional part

    paint pixel (x, y0_int)   with alpha = (1 - frac)
    paint pixel (x, y0_int+1) with alpha = frac
```

Blending formula:
```
new_pixel = old_pixel · (1 - α)  +  color · α
```

Result: smooth, sub-pixel accurate lines with no visible staircase.

---

### 14.3 Bresenham Ellipse (Midpoint Algorithm)

Exploits 4-fold symmetry: compute one quadrant, mirror to all four.

```
# Initialise
x = 0,  y = ry,  rx² = rx·rx,  ry² = ry·ry

# Region 1  (|dy/dx| < 1 — step in x)
d1 = ry² - rx²·ry + 0.25·rx²
while 2·ry²·x < 2·rx²·y:
    plot4(x, y)          ← mirror to all 4 quadrants
    x += 1
    if d1 < 0:  d1 += 2·ry²·x + ry²
    else:       y -= 1;  d1 += 2·ry²·x - 2·rx²·y + ry²

# Region 2  (|dy/dx| > 1 — step in y)
d2 = ry²·(x+0.5)² + rx²·(y-1)² - rx²·ry²
while y >= 0:
    plot4(x, y)
    y -= 1
    if d2 > 0:  d2 += rx² - 2·rx²·y
    else:       x += 1;  d2 += 2·ry²·x - 2·rx²·y + rx²
```

**Filled ellipse** uses the vectorised equation mask (no loop):
```python
mask = ((cc - cx)/rx)**2 + ((rr - cy)/ry)**2 <= 1.0
canvas[mask] = color
```

---

### 14.4 Polygon Scanline Fill

For each row `y` in the bounding box:
```
Find all x-intersections of horizontal line y with polygon edges:

    For each edge (x0,y0)→(x1,y1):
        if min(y0,y1) <= y < max(y0,y1):
            xi = x0 + (y - y0) · (x1 - x0) / (y1 - y0)
            add xi to intersections

Sort intersections.
Fill between pairs: [xi[0], xi[1]], [xi[2], xi[3]], ...  (even-odd rule)
```

**Even-odd rule:** a point is inside the polygon if a ray from it crosses an odd number of edges — correctly handles concave and self-intersecting polygons.

The loop over rows is justified: each row's fill range depends on geometry; the actual fill (slice assignment) is vectorised.

---

*End of Math & Algorithms Notes*
