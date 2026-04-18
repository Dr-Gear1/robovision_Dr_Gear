"""
robovision/utils/drawing_primitives.py
========================================
Drawing primitives implemented directly on NumPy arrays.

All functions draw **in-place** on the canvas and also return it,
allowing method-chaining style usage.

Public API
----------
draw_point          — single pixel or filled circle
draw_line           — Bresenham integer line
draw_line_aa        — Wu anti-aliased line
draw_rectangle      — filled or outline rectangle
draw_polygon        — outline or filled polygon (scanline fill)
draw_ellipse        — filled or outline ellipse (Bresenham)

Shared conventions
------------------
* canvas : np.ndarray shape (H, W) for grayscale or (H, W, 3) for RGB.
* color  : scalar (e.g. 0.8) for grayscale, 3-tuple (R,G,B) for RGB.
           Values in [0, 1] for float canvas, [0, 255] for uint8.
* thickness : int >= 1.  thickness=1 draws a 1-pixel-wide line/outline.
              Thicker strokes are drawn by offsetting the primitive ±t//2
              pixels perpendicular to the stroke direction.
* All coordinates are clipped to canvas boundaries — drawing outside
  the canvas silently clips; no exception is raised for out-of-bounds coords.

Only NumPy — no OpenCV, no PIL, no Matplotlib drawing.
"""

from __future__ import annotations
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# Shared internal utilities
# ══════════════════════════════════════════════════════════════════════════════

def _validate_canvas(canvas: np.ndarray) -> None:
    """Raise TypeError / ValueError for invalid canvas inputs."""
    if not isinstance(canvas, np.ndarray):
        raise TypeError(
            f"'canvas' must be numpy.ndarray, got {type(canvas).__name__}."
        )
    if canvas.ndim == 2:
        return   # grayscale — valid
    if canvas.ndim == 3 and canvas.shape[2] == 3:
        return   # RGB — valid
    raise ValueError(
        f"'canvas' must be shape (H, W) or (H, W, 3), got {canvas.shape}."
    )


def _validate_color(color, canvas: np.ndarray) -> np.ndarray:
    """
    Validate and normalise *color* to match canvas dtype and shape.

    Returns a 1-D numpy array:
      - Grayscale canvas → shape (1,)
      - RGB canvas       → shape (3,)
    """
    is_rgb = canvas.ndim == 3

    if is_rgb:
        if isinstance(color, (int, float)):
            # Scalar → replicate across 3 channels
            c = float(color)
            arr = np.array([c, c, c], dtype=canvas.dtype)
        elif isinstance(color, (tuple, list, np.ndarray)):
            arr = np.asarray(color, dtype=canvas.dtype)
            if arr.shape != (3,):
                raise ValueError(
                    f"For an RGB canvas, 'color' must be a 3-element tuple, "
                    f"got {arr.shape}."
                )
        else:
            raise TypeError(
                f"'color' must be a scalar or (R,G,B) tuple, "
                f"got {type(color).__name__}."
            )
    else:
        if not isinstance(color, (int, float)):
            raise TypeError(
                f"For a grayscale canvas, 'color' must be a scalar, "
                f"got {type(color).__name__}."
            )
        arr = np.array([float(color)], dtype=canvas.dtype)

    return arr


def _validate_thickness(thickness: int) -> int:
    if not isinstance(thickness, int):
        raise TypeError(
            f"'thickness' must be int, got {type(thickness).__name__}."
        )
    if thickness < 1:
        raise ValueError(
            f"'thickness' must be >= 1, got {thickness}."
        )
    return thickness


def _set_pixel(canvas: np.ndarray, r: int, c: int, color_arr: np.ndarray) -> None:
    """Set pixel (r,c) if inside canvas bounds — clip silently."""
    H, W = canvas.shape[:2]
    if 0 <= r < H and 0 <= c < W:
        if canvas.ndim == 2:
            canvas[r, c] = color_arr[0]
        else:
            canvas[r, c] = color_arr


def _set_pixel_alpha(
    canvas: np.ndarray, r: int, c: int,
    color_arr: np.ndarray, alpha: float
) -> None:
    """Blend pixel (r,c) with *alpha* for anti-aliased drawing."""
    H, W = canvas.shape[:2]
    if 0 <= r < H and 0 <= c < W:
        if canvas.ndim == 2:
            canvas[r, c] = canvas[r, c] * (1 - alpha) + color_arr[0] * alpha
        else:
            canvas[r, c] = canvas[r, c] * (1 - alpha) + color_arr * alpha


# ══════════════════════════════════════════════════════════════════════════════
# Point
# ══════════════════════════════════════════════════════════════════════════════

def draw_point(
    canvas: np.ndarray,
    x: int,
    y: int,
    color,
    radius: int = 0,
) -> np.ndarray:
    """
    Draw a single point (pixel) or a filled circle on the canvas.

    Parameters
    ----------
    canvas : np.ndarray
        Image array, shape (H, W) or (H, W, 3), any numeric dtype.
        Modified in place.
    x : int
        Column (horizontal) coordinate.
    y : int
        Row (vertical) coordinate.
    color : scalar or (R, G, B) tuple
        Pixel colour.  Scalar for grayscale canvas, 3-tuple for RGB.
        Values should match the canvas dtype range ([0,1] for float,
        [0,255] for uint8).
    radius : int, optional
        If 0 (default), draw a single pixel.
        If > 0, fill a circle of that radius using a vectorised mask.

    Returns
    -------
    np.ndarray
        The modified canvas (same object, drawn in place).

    Raises
    ------
    TypeError
        If canvas is not ndarray, or color has wrong type.
    ValueError
        If canvas has wrong shape.

    Notes
    -----
    * Coordinates outside the canvas are silently clipped.
    * radius=0 sets exactly one pixel.
    * radius>0 uses a vectorised boolean mask — no Python loops.

    Examples
    --------
    >>> draw_point(canvas, x=100, y=80, color=(1,0,0), radius=3)
    """
    _validate_canvas(canvas)
    color_arr = _validate_color(color, canvas)
    H, W = canvas.shape[:2]

    if radius == 0:
        _set_pixel(canvas, int(y), int(x), color_arr)
    else:
        # Vectorised circle fill using meshgrid mask
        r0, r1 = max(0, int(y) - radius), min(H, int(y) + radius + 1)
        c0, c1 = max(0, int(x) - radius), min(W, int(x) + radius + 1)
        rr, cc = np.ogrid[r0:r1, c0:c1]
        mask   = (rr - int(y)) ** 2 + (cc - int(x)) ** 2 <= radius ** 2
        if canvas.ndim == 2:
            canvas[r0:r1, c0:c1][mask] = color_arr[0]
        else:
            canvas[r0:r1, c0:c1][mask] = color_arr

    return canvas


# ══════════════════════════════════════════════════════════════════════════════
# Line — Bresenham integer
# ══════════════════════════════════════════════════════════════════════════════

def draw_line(
    canvas: np.ndarray,
    x0: int, y0: int,
    x1: int, y1: int,
    color,
    thickness: int = 1,
) -> np.ndarray:
    """
    Draw a straight line using Bresenham's line algorithm.

    Bresenham's algorithm generates all integer pixel coordinates on the
    line using only integer arithmetic (additions and comparisons).

    Algorithm
    ---------
    The classic Bresenham walk:
    1. Compute dx = |x1-x0|, dy = |y1-y0| and step directions.
    2. Initialise error term err = dx - dy.
    3. At each step, set the pixel, then adjust err to decide whether to
       step in x, y, or both.

    Loop justification
    ------------------
    Bresenham's line traversal is inherently sequential — each pixel's
    position depends on the error accumulated so far.  A fully vectorised
    alternative (np.linspace) is used for thickness > 1, drawing parallel
    offset lines.

    Parameters
    ----------
    canvas : np.ndarray
        Image array, shape (H, W) or (H, W, 3).  Modified in place.
    x0, y0 : int
        Start point (column, row).
    x1, y1 : int
        End point (column, row).
    color : scalar or (R, G, B) tuple
        Line colour.
    thickness : int, optional
        Line width in pixels.  Default 1.
        For thickness > 1, parallel Bresenham lines are drawn.

    Returns
    -------
    np.ndarray
        Modified canvas.

    Raises
    ------
    TypeError / ValueError
        See shared validation.

    Notes
    -----
    * For smooth (anti-aliased) lines, use :func:`draw_line_aa`.
    * Coordinates outside canvas bounds are silently clipped per pixel.

    Examples
    --------
    >>> draw_line(canvas, 10, 10, 200, 150, color=(1,0,0), thickness=2)
    """
    _validate_canvas(canvas)
    _validate_thickness(thickness)
    color_arr = _validate_color(color, canvas)

    def _bresenham(x0, y0, x1, y1):
        """Core Bresenham — yields (row, col) pixel coordinates."""
        dx =  abs(x1 - x0);  dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            yield y0, x0
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy;  x0 += sx
            if e2 <= dx:
                err += dx;  y0 += sy

    half = thickness // 2
    # For thickness 1 — single Bresenham pass
    if thickness == 1:
        for r, c in _bresenham(x0, y0, x1, y1):
            _set_pixel(canvas, r, c, color_arr)
    else:
        # Perpendicular offset direction — vectorised
        dx = x1 - x0;  dy = y1 - y0
        length = max(np.hypot(dx, dy), 1e-6)
        px =  dy / length   # perpendicular x
        py = -dx / length   # perpendicular y
        for t in range(-half, half + 1):
            ox = int(round(t * px))
            oy = int(round(t * py))
            for r, c in _bresenham(x0 + ox, y0 + oy, x1 + ox, y1 + oy):
                _set_pixel(canvas, r, c, color_arr)

    return canvas


# ══════════════════════════════════════════════════════════════════════════════
# Anti-aliased line — Xiaolin Wu
# ══════════════════════════════════════════════════════════════════════════════

def draw_line_aa(
    canvas: np.ndarray,
    x0: float, y0: float,
    x1: float, y1: float,
    color,
    thickness: int = 1,
) -> np.ndarray:
    """
    Draw an anti-aliased line using Xiaolin Wu's algorithm.

    Unlike Bresenham (integer-only), Wu's algorithm blends the colour of
    each pixel proportionally to how much the ideal line covers it.  The
    result looks smooth at sub-pixel level without jagged staircasing.

    Algorithm
    ---------
    For each column (or row for steep lines):
    1. Compute the exact sub-pixel y position of the line.
    2. Split into integer part (floor) and fractional part.
    3. Paint the two bracketing pixels with alpha = (1-frac) and frac.

    This is O(max(|dx|, |dy|)) — one pass along the dominant axis.

    Parameters
    ----------
    canvas : np.ndarray
        Image array, shape (H, W) or (H, W, 3), **float** dtype.
        Anti-aliasing requires float blending; integer canvases are
        converted temporarily if needed.
    x0, y0 : float
        Start point (column, row).  Sub-pixel coordinates accepted.
    x1, y1 : float
        End point (column, row).
    color : scalar or (R, G, B) tuple
        Line colour.
    thickness : int, optional
        Line width.  Default 1.

    Returns
    -------
    np.ndarray
        Modified canvas (float32 if canvas was float, original dtype cast back).

    Raises
    ------
    TypeError / ValueError
        See shared validation.

    Notes
    -----
    * For integer canvases, values are temporarily cast to float32 for
      blending and rounded back.
    * Thick AA lines draw multiple parallel Wu passes.

    Examples
    --------
    >>> draw_line_aa(canvas, 10.5, 10.0, 200.5, 150.0, color=0.9)
    """
    _validate_canvas(canvas)
    _validate_thickness(thickness)
    color_arr = _validate_color(color, canvas)

    orig_dtype = canvas.dtype
    if not np.issubdtype(orig_dtype, np.floating):
        canvas[:] = canvas.astype(np.float32)

    def _ipart(x): return int(x)
    def _fpart(x): return x - int(x)
    def _rfpart(x): return 1.0 - _fpart(x)

    def _wu_pass(x0, y0, x1, y1):
        steep = abs(y1 - y0) > abs(x1 - x0)
        if steep:
            x0, y0 = y0, x0
            x1, y1 = y1, x1
        if x0 > x1:
            x0, x1 = x1, x0
            y0, y1 = y1, y0
        dx = x1 - x0
        dy = y1 - y0
        gradient = dy / dx if dx != 0 else 1.0

        # First endpoint
        xend  = round(x0)
        yend  = y0 + gradient * (xend - x0)
        xgap  = _rfpart(x0 + 0.5)
        xpxl1 = int(xend)
        ypxl1 = _ipart(yend)
        if steep:
            _set_pixel_alpha(canvas, xpxl1,   ypxl1,   color_arr, _rfpart(yend) * xgap)
            _set_pixel_alpha(canvas, xpxl1,   ypxl1+1, color_arr, _fpart(yend)  * xgap)
        else:
            _set_pixel_alpha(canvas, ypxl1,   xpxl1,   color_arr, _rfpart(yend) * xgap)
            _set_pixel_alpha(canvas, ypxl1+1, xpxl1,   color_arr, _fpart(yend)  * xgap)

        intery = yend + gradient

        # Second endpoint
        xend  = round(x1)
        yend  = y1 + gradient * (xend - x1)
        xgap  = _fpart(x1 + 0.5)
        xpxl2 = int(xend)
        ypxl2 = _ipart(yend)
        if steep:
            _set_pixel_alpha(canvas, xpxl2,   ypxl2,   color_arr, _rfpart(yend) * xgap)
            _set_pixel_alpha(canvas, xpxl2,   ypxl2+1, color_arr, _fpart(yend)  * xgap)
        else:
            _set_pixel_alpha(canvas, ypxl2,   xpxl2,   color_arr, _rfpart(yend) * xgap)
            _set_pixel_alpha(canvas, ypxl2+1, xpxl2,   color_arr, _fpart(yend)  * xgap)

        # Main loop
        for x in range(xpxl1 + 1, xpxl2):
            if steep:
                _set_pixel_alpha(canvas, x, _ipart(intery),   color_arr, _rfpart(intery))
                _set_pixel_alpha(canvas, x, _ipart(intery)+1, color_arr, _fpart(intery))
            else:
                _set_pixel_alpha(canvas, _ipart(intery),   x, color_arr, _rfpart(intery))
                _set_pixel_alpha(canvas, _ipart(intery)+1, x, color_arr, _fpart(intery))
            intery += gradient

    half = thickness // 2
    dx   = x1 - x0;  dy = y1 - y0
    length = max(np.hypot(dx, dy), 1e-6)
    px =  dy / length;  py = -dx / length

    for t in range(-half, half + 1):
        ox = t * px;  oy = t * py
        _wu_pass(x0 + ox, y0 + oy, x1 + ox, y1 + oy)

    return canvas


# ══════════════════════════════════════════════════════════════════════════════
# Rectangle
# ══════════════════════════════════════════════════════════════════════════════

def draw_rectangle(
    canvas: np.ndarray,
    x0: int, y0: int,
    x1: int, y1: int,
    color,
    filled: bool = False,
    thickness: int = 1,
) -> np.ndarray:
    """
    Draw a rectangle — filled or outline — on the canvas.

    The rectangle spans columns [x0, x1] and rows [y0, y1].
    Coordinates are automatically sorted so x0 ≤ x1 and y0 ≤ y1.

    Parameters
    ----------
    canvas : np.ndarray
        Image array, shape (H, W) or (H, W, 3).  Modified in place.
    x0, y0 : int
        One corner (column, row).
    x1, y1 : int
        Opposite corner (column, row).
    color : scalar or (R, G, B) tuple
        Fill / outline colour.
    filled : bool, optional
        If True, fill the rectangle.  If False (default), draw outline only.
    thickness : int, optional
        Outline thickness in pixels.  Ignored when filled=True.  Default 1.

    Returns
    -------
    np.ndarray
        Modified canvas.

    Raises
    ------
    TypeError / ValueError
        See shared validation.

    Notes
    -----
    * Filled rectangle uses a single NumPy slice assignment — O(area), no loops.
    * Outline draws four lines via :func:`draw_line`.
    * Clipped to canvas bounds automatically.

    Examples
    --------
    >>> draw_rectangle(canvas, 50, 50, 200, 150, color=(0,1,0), filled=True)
    >>> draw_rectangle(canvas, 50, 50, 200, 150, color=(1,0,0), thickness=2)
    """
    _validate_canvas(canvas)
    _validate_thickness(thickness)
    color_arr = _validate_color(color, canvas)

    H, W = canvas.shape[:2]
    x0, x1 = sorted([int(x0), int(x1)])
    y0, y1 = sorted([int(y0), int(y1)])

    if filled:
        # Clip to canvas then assign — fully vectorised
        r0 = max(0, y0); r1 = min(H, y1 + 1)
        c0 = max(0, x0); c1 = min(W, x1 + 1)
        if r0 < r1 and c0 < c1:
            if canvas.ndim == 2:
                canvas[r0:r1, c0:c1] = color_arr[0]
            else:
                canvas[r0:r1, c0:c1] = color_arr
    else:
        # Four border lines via draw_line
        draw_line(canvas, x0, y0, x1, y0, color, thickness)   # top
        draw_line(canvas, x0, y1, x1, y1, color, thickness)   # bottom
        draw_line(canvas, x0, y0, x0, y1, color, thickness)   # left
        draw_line(canvas, x1, y0, x1, y1, color, thickness)   # right

    return canvas


# ══════════════════════════════════════════════════════════════════════════════
# Polygon
# ══════════════════════════════════════════════════════════════════════════════

def draw_polygon(
    canvas: np.ndarray,
    points: list[tuple[int, int]],
    color,
    filled: bool = False,
    thickness: int = 1,
) -> np.ndarray:
    """
    Draw a polygon — outline or filled — on the canvas.

    Parameters
    ----------
    canvas : np.ndarray
        Image array, shape (H, W) or (H, W, 3).  Modified in place.
    points : list of (x, y) tuples
        Vertex coordinates as (column, row) pairs.  At least 2 points
        required for a line; at least 3 for a polygon.
    color : scalar or (R, G, B) tuple
        Fill / outline colour.
    filled : bool, optional
        If True, fill using scanline algorithm.  Default False.
    thickness : int, optional
        Edge thickness (outline only).  Default 1.

    Returns
    -------
    np.ndarray
        Modified canvas.

    Raises
    ------
    TypeError
        If canvas is not ndarray or points is not a list/tuple.
    ValueError
        If fewer than 2 points provided.

    Notes
    -----
    * Outline draws each edge with :func:`draw_line`.
    * Fill uses a scanline algorithm — for each row in the bounding box,
      finds the x-intersections of the polygon edges and fills between them.
      The scanline loop (over rows) is justified because each row's
      fill range depends on edge geometry; the fill itself is vectorised.
    * Self-intersecting polygons fill with even-odd rule.

    Examples
    --------
    >>> pts = [(50,50),(150,30),(200,120),(100,160),(20,100)]
    >>> draw_polygon(canvas, pts, color=(0,0,1), filled=True)
    """
    _validate_canvas(canvas)
    _validate_thickness(thickness)
    color_arr = _validate_color(color, canvas)
    if not isinstance(points, (list, tuple)):
        raise TypeError(
            f"'points' must be a list of (x,y) tuples, got {type(points).__name__}."
        )
    if len(points) < 2:
        raise ValueError(
            f"'points' must have at least 2 vertices, got {len(points)}."
        )

    pts = [(int(p[0]), int(p[1])) for p in points]
    n   = len(pts)

    # ── Outline ───────────────────────────────────────────────────────────
    for i in range(n):
        x0, y0 = pts[i]
        x1, y1 = pts[(i + 1) % n]
        draw_line(canvas, x0, y0, x1, y1, color, thickness)

    # ── Filled (scanline) ─────────────────────────────────────────────────
    if filled and n >= 3:
        H, W   = canvas.shape[:2]
        ys     = [p[1] for p in pts]
        y_min  = max(0, min(ys))
        y_max  = min(H - 1, max(ys))

        # Scanline loop — one iteration per row in the bounding box
        # Justified: each row's fill segment depends on edge intersections
        # which cannot be precomputed without iterating over rows.
        # The fill itself (canvas slice) is vectorised.
        for y in range(y_min, y_max + 1):
            x_intersects = []
            for i in range(n):
                x0, y0 = pts[i]
                x1, y1 = pts[(i + 1) % n]
                if min(y0, y1) <= y < max(y0, y1):
                    if y1 != y0:
                        xi = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
                        x_intersects.append(xi)
            x_intersects.sort()
            # Fill between pairs of intersections (even-odd rule)
            for k in range(0, len(x_intersects) - 1, 2):
                xa = max(0, int(np.ceil(x_intersects[k])))
                xb = min(W,  int(np.floor(x_intersects[k + 1])) + 1)
                if xa < xb:
                    if canvas.ndim == 2:
                        canvas[y, xa:xb] = color_arr[0]
                    else:
                        canvas[y, xa:xb] = color_arr

    return canvas


# ══════════════════════════════════════════════════════════════════════════════
# Ellipse — Bresenham midpoint algorithm
# ══════════════════════════════════════════════════════════════════════════════

def draw_ellipse(
    canvas: np.ndarray,
    cx: int, cy: int,
    rx: int, ry: int,
    color,
    filled: bool = False,
    thickness: int = 1,
) -> np.ndarray:
    """
    Draw an ellipse — filled or outline — using Bresenham's midpoint method.

    The ellipse equation: (x - cx)²/rx² + (y - cy)²/ry² = 1

    Algorithm
    ---------
    Bresenham's midpoint ellipse algorithm generates pixel coordinates
    in quadrant I and mirrors them to all four quadrants, exploiting
    the 4-fold symmetry of an axis-aligned ellipse.

    The algorithm walks along the ellipse boundary:
    - Region 1: |dy/dx| < 1  (slope < 1, step in x)
    - Region 2: |dy/dx| > 1  (slope > 1, step in y)

    Loop justification
    ------------------
    The midpoint ellipse walk is inherently sequential — each step's
    direction depends on the decision parameter accumulated so far.
    The loop count is O(rx + ry) — not O(H × W).

    Parameters
    ----------
    canvas : np.ndarray
        Image array, shape (H, W) or (H, W, 3).  Modified in place.
    cx, cy : int
        Centre pixel (column, row).
    rx, ry : int
        Semi-axes in x (horizontal) and y (vertical) directions.
        Both must be >= 1.
    color : scalar or (R, G, B) tuple
        Ellipse colour.
    filled : bool, optional
        If True, fill the ellipse using a vectorised mask.  Default False.
    thickness : int, optional
        Outline thickness.  Ignored when filled=True.  Default 1.

    Returns
    -------
    np.ndarray
        Modified canvas.

    Raises
    ------
    TypeError / ValueError
        See shared validation.  Also raises if rx < 1 or ry < 1.

    Notes
    -----
    * Filled ellipse uses ``np.ogrid`` boolean mask — fully vectorised, O(rx*ry).
    * Circle: set rx == ry.

    Examples
    --------
    >>> draw_ellipse(canvas, cx=128, cy=128, rx=80, ry=50, color=(1,0.5,0))
    >>> draw_ellipse(canvas, 128, 128, 60, 60, color=0.9, filled=True)  # circle
    """
    _validate_canvas(canvas)
    _validate_thickness(thickness)
    color_arr = _validate_color(color, canvas)
    cx, cy, rx, ry = int(cx), int(cy), int(rx), int(ry)
    if rx < 1 or ry < 1:
        raise ValueError(
            f"'rx' and 'ry' must each be >= 1, got rx={rx}, ry={ry}."
        )

    H, W = canvas.shape[:2]

    # ── Filled — vectorised mask ───────────────────────────────────────────
    if filled:
        r0 = max(0, cy - ry); r1 = min(H, cy + ry + 1)
        c0 = max(0, cx - rx); c1 = min(W, cx + rx + 1)
        rr, cc = np.ogrid[r0:r1, c0:c1]
        mask   = ((cc - cx) / rx) ** 2 + ((rr - cy) / ry) ** 2 <= 1.0
        if canvas.ndim == 2:
            canvas[r0:r1, c0:c1][mask] = color_arr[0]
        else:
            canvas[r0:r1, c0:c1][mask] = color_arr
        return canvas

    # ── Outline — Bresenham midpoint ──────────────────────────────────────
    def _plot4(x, y):
        """Paint the 4 symmetric ellipse pixels with thickness."""
        half = thickness // 2
        for dr in range(-half, half + 1):
            for dc in range(-half, half + 1):
                _set_pixel(canvas, cy + y + dr, cx + x + dc, color_arr)
                _set_pixel(canvas, cy - y + dr, cx + x + dc, color_arr)
                _set_pixel(canvas, cy + y + dr, cx - x + dc, color_arr)
                _set_pixel(canvas, cy - y + dr, cx - x + dc, color_arr)

    rx2 = rx * rx;  ry2 = ry * ry
    x   = 0;        y   = ry

    # Region 1
    d1 = ry2 - rx2 * ry + 0.25 * rx2
    dx = 2 * ry2 * x
    dy = 2 * rx2 * y
    while dx < dy:
        _plot4(x, y)
        x += 1;  dx += 2 * ry2
        if d1 < 0:
            d1 += dx + ry2
        else:
            y -= 1;  dy -= 2 * rx2;  d1 += dx - dy + ry2

    # Region 2
    d2 = ry2 * (x + 0.5) ** 2 + rx2 * (y - 1) ** 2 - rx2 * ry2
    while y >= 0:
        _plot4(x, y)
        y -= 1;  dy -= 2 * rx2
        if d2 > 0:
            d2 += rx2 - dy
        else:
            x += 1;  dx += 2 * ry2;  d2 += dx - dy + rx2

    return canvas
