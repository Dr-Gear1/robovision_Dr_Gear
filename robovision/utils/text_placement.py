"""
robovision/utils/text_placement.py
=====================================
Text placement on NumPy image arrays using a built-in bitmap font.

Renders text WITHOUT any external font libraries (no PIL, no cv2, no freetype).
Uses a compact 5×7 pixel bitmap font embedded directly in this module.

Public API
----------
draw_text   — render a string at position (x, y) with scale and colour
get_text_size — compute bounding box of a string before drawing

Font details
------------
* Built-in 5×7 bitmap font covering ASCII 32–126 (space through tilde).
* Each character is a list of 7 integers; each bit in an integer represents
  one pixel column (MSB = leftmost column of a 5-pixel-wide glyph).
* Scale > 1 enlarges each pixel by an integer factor (nearest-neighbour),
  so scale=2 gives 10×14 px glyphs, scale=3 gives 15×21 px, etc.

Only NumPy — no PIL, no freetype, no cv2, no Matplotlib fonts.
"""

from __future__ import annotations
import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# Embedded 5×7 bitmap font  (ASCII 32 – 126)
# ══════════════════════════════════════════════════════════════════════════════
# Each entry: list of 7 integers.
# Each integer encodes one row of 5 pixels (bit 4 = leftmost pixel).
# 0b10000 = pixel at column 0, 0b00001 = pixel at column 4.

_FONT_5X7: dict[int, list[int]] = {
    32:  [0x00,0x00,0x00,0x00,0x00,0x00,0x00],  # space
    33:  [0x04,0x04,0x04,0x04,0x00,0x00,0x04],  # !
    34:  [0x0A,0x0A,0x00,0x00,0x00,0x00,0x00],  # "
    35:  [0x0A,0x0A,0x1F,0x0A,0x1F,0x0A,0x0A],  # #
    36:  [0x04,0x0F,0x14,0x0E,0x05,0x1E,0x04],  # $
    37:  [0x18,0x19,0x02,0x04,0x08,0x13,0x03],  # %
    38:  [0x0C,0x12,0x14,0x08,0x15,0x12,0x0D],  # &
    39:  [0x04,0x04,0x00,0x00,0x00,0x00,0x00],  # '
    40:  [0x02,0x04,0x08,0x08,0x08,0x04,0x02],  # (
    41:  [0x08,0x04,0x02,0x02,0x02,0x04,0x08],  # )
    42:  [0x00,0x04,0x15,0x0E,0x15,0x04,0x00],  # *
    43:  [0x00,0x04,0x04,0x1F,0x04,0x04,0x00],  # +
    44:  [0x00,0x00,0x00,0x00,0x06,0x04,0x08],  # ,
    45:  [0x00,0x00,0x00,0x1F,0x00,0x00,0x00],  # -
    46:  [0x00,0x00,0x00,0x00,0x00,0x06,0x06],  # .
    47:  [0x00,0x01,0x02,0x04,0x08,0x10,0x00],  # /
    48:  [0x0E,0x11,0x13,0x15,0x19,0x11,0x0E],  # 0
    49:  [0x04,0x0C,0x04,0x04,0x04,0x04,0x0E],  # 1
    50:  [0x0E,0x11,0x01,0x02,0x04,0x08,0x1F],  # 2
    51:  [0x1F,0x02,0x04,0x02,0x01,0x11,0x0E],  # 3
    52:  [0x02,0x06,0x0A,0x12,0x1F,0x02,0x02],  # 4
    53:  [0x1F,0x10,0x1E,0x01,0x01,0x11,0x0E],  # 5
    54:  [0x06,0x08,0x10,0x1E,0x11,0x11,0x0E],  # 6
    55:  [0x1F,0x01,0x02,0x04,0x08,0x08,0x08],  # 7
    56:  [0x0E,0x11,0x11,0x0E,0x11,0x11,0x0E],  # 8
    57:  [0x0E,0x11,0x11,0x0F,0x01,0x02,0x0C],  # 9
    58:  [0x00,0x06,0x06,0x00,0x06,0x06,0x00],  # :
    59:  [0x00,0x06,0x06,0x00,0x06,0x04,0x08],  # ;
    60:  [0x02,0x04,0x08,0x10,0x08,0x04,0x02],  # <
    61:  [0x00,0x00,0x1F,0x00,0x1F,0x00,0x00],  # =
    62:  [0x08,0x04,0x02,0x01,0x02,0x04,0x08],  # >
    63:  [0x0E,0x11,0x01,0x02,0x04,0x00,0x04],  # ?
    64:  [0x0E,0x11,0x01,0x0D,0x15,0x15,0x0E],  # @
    65:  [0x04,0x0A,0x11,0x11,0x1F,0x11,0x11],  # A
    66:  [0x1E,0x11,0x11,0x1E,0x11,0x11,0x1E],  # B
    67:  [0x0E,0x11,0x10,0x10,0x10,0x11,0x0E],  # C
    68:  [0x1C,0x12,0x11,0x11,0x11,0x12,0x1C],  # D
    69:  [0x1F,0x10,0x10,0x1E,0x10,0x10,0x1F],  # E
    70:  [0x1F,0x10,0x10,0x1E,0x10,0x10,0x10],  # F
    71:  [0x0E,0x11,0x10,0x17,0x11,0x11,0x0F],  # G
    72:  [0x11,0x11,0x11,0x1F,0x11,0x11,0x11],  # H
    73:  [0x0E,0x04,0x04,0x04,0x04,0x04,0x0E],  # I
    74:  [0x07,0x02,0x02,0x02,0x02,0x12,0x0C],  # J
    75:  [0x11,0x12,0x14,0x18,0x14,0x12,0x11],  # K
    76:  [0x10,0x10,0x10,0x10,0x10,0x10,0x1F],  # L
    77:  [0x11,0x1B,0x15,0x15,0x11,0x11,0x11],  # M
    78:  [0x11,0x11,0x19,0x15,0x13,0x11,0x11],  # N
    79:  [0x0E,0x11,0x11,0x11,0x11,0x11,0x0E],  # O
    80:  [0x1E,0x11,0x11,0x1E,0x10,0x10,0x10],  # P
    81:  [0x0E,0x11,0x11,0x11,0x15,0x12,0x0D],  # Q
    82:  [0x1E,0x11,0x11,0x1E,0x14,0x12,0x11],  # R
    83:  [0x0F,0x10,0x10,0x0E,0x01,0x01,0x1E],  # S
    84:  [0x1F,0x04,0x04,0x04,0x04,0x04,0x04],  # T
    85:  [0x11,0x11,0x11,0x11,0x11,0x11,0x0E],  # U
    86:  [0x11,0x11,0x11,0x11,0x11,0x0A,0x04],  # V
    87:  [0x11,0x11,0x15,0x15,0x15,0x15,0x0A],  # W
    88:  [0x11,0x11,0x0A,0x04,0x0A,0x11,0x11],  # X
    89:  [0x11,0x11,0x0A,0x04,0x04,0x04,0x04],  # Y
    90:  [0x1F,0x01,0x02,0x04,0x08,0x10,0x1F],  # Z
    91:  [0x0E,0x08,0x08,0x08,0x08,0x08,0x0E],  # [
    92:  [0x00,0x10,0x08,0x04,0x02,0x01,0x00],  # backslash
    93:  [0x0E,0x02,0x02,0x02,0x02,0x02,0x0E],  # ]
    94:  [0x04,0x0A,0x11,0x00,0x00,0x00,0x00],  # ^
    95:  [0x00,0x00,0x00,0x00,0x00,0x00,0x1F],  # _
    96:  [0x08,0x04,0x00,0x00,0x00,0x00,0x00],  # `
    97:  [0x00,0x00,0x0E,0x01,0x0F,0x11,0x0F],  # a
    98:  [0x10,0x10,0x1E,0x11,0x11,0x11,0x1E],  # b
    99:  [0x00,0x00,0x0E,0x10,0x10,0x11,0x0E],  # c
   100:  [0x01,0x01,0x0F,0x11,0x11,0x11,0x0F],  # d
   101:  [0x00,0x00,0x0E,0x11,0x1F,0x10,0x0E],  # e
   102:  [0x06,0x09,0x08,0x1C,0x08,0x08,0x08],  # f
   103:  [0x00,0x0F,0x11,0x11,0x0F,0x01,0x0E],  # g
   104:  [0x10,0x10,0x1E,0x11,0x11,0x11,0x11],  # h
   105:  [0x04,0x00,0x0C,0x04,0x04,0x04,0x0E],  # i
   106:  [0x02,0x00,0x06,0x02,0x02,0x12,0x0C],  # j
   107:  [0x10,0x10,0x12,0x14,0x18,0x14,0x12],  # k
   108:  [0x0C,0x04,0x04,0x04,0x04,0x04,0x0E],  # l
   109:  [0x00,0x00,0x1A,0x15,0x15,0x11,0x11],  # m
   110:  [0x00,0x00,0x1E,0x11,0x11,0x11,0x11],  # n
   111:  [0x00,0x00,0x0E,0x11,0x11,0x11,0x0E],  # o
   112:  [0x00,0x00,0x1E,0x11,0x11,0x1E,0x10],  # p
   113:  [0x00,0x00,0x0F,0x11,0x11,0x0F,0x01],  # q
   114:  [0x00,0x00,0x16,0x19,0x10,0x10,0x10],  # r
   115:  [0x00,0x00,0x0F,0x10,0x0E,0x01,0x1E],  # s
   116:  [0x08,0x08,0x1C,0x08,0x08,0x09,0x06],  # t
   117:  [0x00,0x00,0x11,0x11,0x11,0x13,0x0D],  # u
   118:  [0x00,0x00,0x11,0x11,0x11,0x0A,0x04],  # v
   119:  [0x00,0x00,0x11,0x15,0x15,0x15,0x0A],  # w
   120:  [0x00,0x00,0x11,0x0A,0x04,0x0A,0x11],  # x
   121:  [0x00,0x00,0x11,0x11,0x0F,0x01,0x0E],  # y
   122:  [0x00,0x00,0x1F,0x02,0x04,0x08,0x1F],  # z
   123:  [0x06,0x08,0x08,0x10,0x08,0x08,0x06],  # {
   124:  [0x04,0x04,0x04,0x00,0x04,0x04,0x04],  # |
   125:  [0x0C,0x02,0x02,0x01,0x02,0x02,0x0C],  # }
   126:  [0x08,0x15,0x02,0x00,0x00,0x00,0x00],  # ~
}

_GLYPH_W = 5   # glyph width  in pixels at scale 1
_GLYPH_H = 7   # glyph height in pixels at scale 1
_GLYPH_GAP = 1  # pixels between characters at scale 1


# ══════════════════════════════════════════════════════════════════════════════
# Validation helpers
# ══════════════════════════════════════════════════════════════════════════════

def _validate_canvas(canvas: np.ndarray) -> None:
    if not isinstance(canvas, np.ndarray):
        raise TypeError(
            f"'canvas' must be numpy.ndarray, got {type(canvas).__name__}."
        )
    if canvas.ndim == 2:
        return
    if canvas.ndim == 3 and canvas.shape[2] == 3:
        return
    raise ValueError(
        f"'canvas' must be shape (H, W) or (H, W, 3), got {canvas.shape}."
    )


def _validate_color(color, canvas: np.ndarray) -> np.ndarray:
    is_rgb = canvas.ndim == 3
    if is_rgb:
        if isinstance(color, (int, float)):
            c = float(color)
            return np.array([c, c, c], dtype=canvas.dtype)
        arr = np.asarray(color, dtype=canvas.dtype)
        if arr.shape != (3,):
            raise ValueError(
                f"For RGB canvas, color must be a 3-element tuple, got {arr.shape}."
            )
        return arr
    else:
        if not isinstance(color, (int, float)):
            raise TypeError(
                f"For grayscale canvas, color must be scalar, "
                f"got {type(color).__name__}."
            )
        return np.array([float(color)], dtype=canvas.dtype)


# ══════════════════════════════════════════════════════════════════════════════
# Public: get_text_size
# ══════════════════════════════════════════════════════════════════════════════

def get_text_size(text: str, scale: int = 1) -> tuple[int, int]:
    """
    Compute the bounding box (width, height) of a text string.

    Useful for aligning or centring text before drawing it.

    Parameters
    ----------
    text : str
        The string to measure.
    scale : int, optional
        Integer scale factor.  Default 1 (5×7 px per character).

    Returns
    -------
    (width, height) : tuple of int
        Pixel dimensions of the rendered text at the given scale.
        width  = len(text) * (GLYPH_W + GAP) * scale − GAP * scale
        height = GLYPH_H * scale

    Raises
    ------
    TypeError
        If text is not str, or scale is not int.
    ValueError
        If scale < 1.

    Examples
    --------
    >>> w, h = get_text_size("Hello", scale=2)
    >>> print(w, h)   # 58, 14
    """
    if not isinstance(text, str):
        raise TypeError(f"'text' must be str, got {type(text).__name__}.")
    if not isinstance(scale, int):
        raise TypeError(f"'scale' must be int, got {type(scale).__name__}.")
    if scale < 1:
        raise ValueError(f"'scale' must be >= 1, got {scale}.")
    if not text:
        return (0, _GLYPH_H * scale)

    n    = len(text)
    w    = (n * (_GLYPH_W + _GLYPH_GAP) - _GLYPH_GAP) * scale
    h    = _GLYPH_H * scale
    return (w, h)


# ══════════════════════════════════════════════════════════════════════════════
# Public: draw_text
# ══════════════════════════════════════════════════════════════════════════════

def draw_text(
    canvas: np.ndarray,
    text: str,
    x: int,
    y: int,
    color,
    scale: int = 1,
    background_color=None,
) -> np.ndarray:
    """
    Render a text string onto a NumPy image array.

    Uses the built-in 5×7 bitmap font.  Each character glyph is
    upscaled by *scale* using ``np.repeat`` (nearest-neighbour, no blur).

    Font specifications
    -------------------
    * Glyph size at scale 1 : 5 × 7 pixels (width × height)
    * Character spacing     : 1 pixel gap between glyphs
    * Supports ASCII 32–126 (printable characters)
    * Unsupported characters are rendered as a rectangle outline

    Parameters
    ----------
    canvas : np.ndarray
        Image array, shape (H, W) or (H, W, 3), any numeric dtype.
        Modified in place.
    text : str
        String to render.  Any ASCII printable character is supported.
        Non-printable or out-of-range characters are skipped gracefully.
    x : int
        Left edge of the text bounding box, in pixels (column index).
    y : int
        Top edge of the text bounding box, in pixels (row index).
    color : scalar or (R, G, B) tuple
        Text colour.  Scalar for grayscale canvas, 3-tuple for RGB.
        Values should match the canvas dtype range
        ([0.0, 1.0] for float, [0, 255] for uint8).
    scale : int, optional
        Integer upscale factor.  Default 1.
        scale=1 → 5×7 px glyphs, scale=2 → 10×14 px, scale=3 → 15×21 px.
        Non-integer scales are not supported (use scale >= 1).
    background_color : scalar / tuple or None, optional
        If not None, fill the text bounding box with this colour before
        rendering the glyphs.  Useful for text on noisy backgrounds.
        Default None (no background fill).

    Returns
    -------
    np.ndarray
        The modified canvas (same object, drawn in place).

    Raises
    ------
    TypeError
        If canvas is not ndarray, text is not str, scale is not int,
        or color has wrong type for the canvas mode.
    ValueError
        If canvas has wrong shape, scale < 1.

    Notes
    -----
    * Text that extends beyond canvas boundaries is silently clipped
      pixel-by-pixel — partial characters at the edge are rendered as far
      as they fit.
    * The upscale uses ``np.repeat`` along both axes — fully vectorised,
      no Python loops per pixel.
    * For very large text, consider scale=3 or scale=4.
      scale=1 is readable at 1× zoom; scale=2 is comfortable for labels.

    Examples
    --------
    >>> canvas = np.zeros((200, 400, 3), dtype=np.float32)
    >>> draw_text(canvas, "Hello RoboVision!", x=10, y=10,
    ...           color=(1.0, 1.0, 0.0), scale=2)

    >>> gray = np.ones((100, 300), dtype=np.float32) * 0.5
    >>> draw_text(gray, "Score: 42", x=5, y=5, color=0.0, scale=1)
    """
    _validate_canvas(canvas)
    if not isinstance(text, str):
        raise TypeError(f"'text' must be str, got {type(text).__name__}.")
    if not isinstance(scale, int) or scale < 1:
        raise ValueError(
            f"'scale' must be a positive int, got {scale!r}."
        )
    color_arr = _validate_color(color, canvas)

    gw  = _GLYPH_W  * scale
    gh  = _GLYPH_H  * scale
    gap = _GLYPH_GAP * scale

    H, W = canvas.shape[:2]

    # ── Optional background fill ───────────────────────────────────────────
    if background_color is not None:
        bg_arr   = _validate_color(background_color, canvas)
        tw, th   = get_text_size(text, scale)
        r0 = max(0, y);      r1 = min(H, y + th)
        c0 = max(0, x);      c1 = min(W, x + tw)
        if r0 < r1 and c0 < c1:
            if canvas.ndim == 2:
                canvas[r0:r1, c0:c1] = bg_arr[0]
            else:
                canvas[r0:r1, c0:c1] = bg_arr

    # ── Render each character ─────────────────────────────────────────────
    cursor_x = x
    for ch in text:
        code = ord(ch)
        rows = _FONT_5X7.get(code)

        if rows is None:
            # Unsupported char — draw a small rectangle placeholder
            if canvas.ndim == 2:
                r0 = max(0, y); r1 = min(H, y + gh)
                c0 = max(0, cursor_x); c1 = min(W, cursor_x + gw)
                canvas[r0:r0+scale, c0:c1] = color_arr[0]
                canvas[r1-scale:r1, c0:c1] = color_arr[0]
                canvas[r0:r1, c0:c0+scale] = color_arr[0]
                canvas[r0:r1, c1-scale:c1] = color_arr[0]
            cursor_x += gw + gap
            continue

        # Build (7, 5) binary glyph array — fully vectorised
        glyph = np.zeros((_GLYPH_H, _GLYPH_W), dtype=np.uint8)
        for row_idx, row_bits in enumerate(rows):
            for col_idx in range(_GLYPH_W):
                if row_bits & (1 << (_GLYPH_W - 1 - col_idx)):
                    glyph[row_idx, col_idx] = 1

        # Upscale via np.repeat — no per-pixel loops
        if scale > 1:
            glyph = np.repeat(np.repeat(glyph, scale, axis=0), scale, axis=1)

        # Blit glyph onto canvas with bounds clipping
        for gr in range(gh):
            cr = y + gr
            if cr < 0 or cr >= H:
                continue
            for gc in range(gw):
                cc = cursor_x + gc
                if cc < 0 or cc >= W:
                    continue
                if glyph[gr, gc]:
                    if canvas.ndim == 2:
                        canvas[cr, cc] = color_arr[0]
                    else:
                        canvas[cr, cc] = color_arr

        cursor_x += gw + gap

    return canvas
