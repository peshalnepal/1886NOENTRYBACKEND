"""A minimal, dependency-free PDF writer.

The backend pins a deliberately small dependency set (see requirements.txt) and
neither reportlab nor weasyprint is available, so this module hand-rolls just
enough of the PDF spec to produce a clean multi-page report:

  - flowing text with automatic word-wrap and page breaks, in the two standard
    base-14 fonts (Helvetica / Helvetica-Bold — no font embedding required);
  - clickable link annotations (URI actions) for snapshot / clip / dashboard URLs;
  - embedded JPEG thumbnails inside table cells (raw JPEG bytes pass through the
    PDF ``DCTDecode`` filter untouched; other formats are transcoded via Pillow).

Coordinates follow the PDF convention (origin bottom-left, points). The public
``PDFReport`` class exposes a top-down cursor model (``y`` measured from the top
margin downward) so callers think in normal document flow.
"""

from __future__ import annotations

import io
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple,Dict

# A4 in points.
PAGE_W = 595.28
PAGE_H = 841.89
MARGIN = 50.0

# Helvetica / Helvetica-Bold advance widths (AFM, 1000-unit em) for ASCII
# 32..126. Used for word-wrap and to size link-annotation rectangles. Indexed
# by ``ord(ch) - 32``.
_HELV = [
    278, 278, 355, 556, 556, 889, 667, 191, 333, 333, 389, 584, 278, 333, 278,
    278, 556, 556, 556, 556, 556, 556, 556, 556, 556, 556, 278, 278, 584, 584,
    584, 556, 1015, 667, 667, 722, 722, 667, 611, 778, 722, 278, 500, 667, 556,
    833, 722, 778, 667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611, 278,
    278, 278, 469, 556, 333, 556, 556, 500, 556, 556, 278, 556, 556, 222, 222,
    500, 222, 833, 556, 556, 556, 556, 333, 500, 278, 556, 500, 722, 500, 500,
    500, 334, 260, 334, 584,
]
_HELV_BOLD = [
    278, 333, 474, 556, 556, 889, 722, 238, 333, 333, 389, 584, 278, 333, 278,
    278, 556, 556, 556, 556, 556, 556, 556, 556, 556, 556, 333, 333, 584, 584,
    584, 611, 975, 722, 722, 722, 722, 667, 611, 778, 722, 278, 556, 722, 611,
    833, 722, 778, 667, 778, 722, 667, 611, 722, 667, 944, 667, 667, 611, 333,
    278, 333, 584, 556, 333, 556, 611, 556, 611, 556, 333, 611, 611, 278, 278,
    556, 278, 889, 611, 611, 611, 611, 389, 556, 333, 611, 556, 778, 556, 556,
    500, 389, 280, 389, 584,
]

Color = Tuple[float, float, float]


def _char_width(ch: str, size: float, bold: bool) -> float:
    code = ord(ch)
    table = _HELV_BOLD if bold else _HELV
    if 32 <= code <= 126:
        w = table[code - 32]
    else:
        w = 556  # fallback for anything outside the metric table
    return (w / 1000.0) * size


def text_width(s: str, size: float, bold: bool = False) -> float:
    return sum(_char_width(ch, size, bold) for ch in s)


# Common typographic characters mapped to their WinAnsiEncoding byte values
# (the fonts are declared with /WinAnsiEncoding). Without this, an em-dash or a
# curly quote would fall outside Latin-1 and render as '?'.
_WINANSI_MAP = {
    "‘": 0x91, "’": 0x92, "“": 0x93, "”": 0x94,
    "•": 0x95, "–": 0x96, "—": 0x97, "…": 0x85,
    "€": 0x80, "™": 0x99, "‹": 0x8B, "›": 0x9B,
}


def _escape_pdf_text(s: str) -> bytes:
    """Escape a string for a PDF literal ``( ... )``, encoded as WinAnsi bytes.

    Common typographic characters are remapped to their WinAnsi code points;
    anything still outside the 32..255 range is replaced with ``?`` so the byte
    stream never desyncs.
    """
    out = bytearray()
    for ch in s:
        if ch in ("\\", "(", ")"):
            out.append(0x5C)  # backslash
            out.append(ord(ch))
        elif 32 <= ord(ch) <= 255:
            out.append(ord(ch))
        elif ch in _WINANSI_MAP:
            out.append(_WINANSI_MAP[ch])
        else:
            out.append(0x3F)  # '?'
    return bytes(out)


# ---------------------------------------------------------------------------
# JPEG handling
# ---------------------------------------------------------------------------
def _jpeg_info(data: bytes) -> Optional[Tuple[int, int, int]]:
    """Return ``(width, height, components)`` for raw JPEG bytes, or None."""
    if len(data) < 4 or data[0] != 0xFF or data[1] != 0xD8:
        return None
    i = 2
    n = len(data)
    while i + 1 < n:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        i += 2
        # Standalone markers (no length): padding, RSTn, SOI/EOI.
        if marker == 0xFF or marker in (0x01, 0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            continue
        if i + 2 > n:
            break
        seg_len = (data[i] << 8) | data[i + 1]
        # SOF0..SOF15 carry the frame geometry; exclude DHT/DAC/DRI markers.
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            if i + 7 < n:
                h = (data[i + 3] << 8) | data[i + 4]
                w = (data[i + 5] << 8) | data[i + 6]
                comps = data[i + 7]
                return (w, h, comps)
            return None
        i += seg_len
    return None


def normalize_to_jpeg(raw: bytes) -> Optional[Tuple[bytes, int, int, int]]:
    """Coerce arbitrary image bytes to embeddable JPEG.

    Returns ``(jpeg_bytes, width, height, components)`` or None when the image
    cannot be decoded (e.g. a non-JPEG with Pillow unavailable).
    """
    if not raw:
        return None

    info = _jpeg_info(raw)
    if info is not None:
        w, h, comps = info
        if w > 0 and h > 0 and comps in (1, 3, 4):
            return raw, w, h, comps

    # Not a (usable) JPEG: transcode with Pillow if we have it.
    try:
        from PIL import Image  # type: ignore
    except Exception:
        return None

    try:
        im = Image.open(io.BytesIO(raw))
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        comps = 1 if im.mode == "L" else 3
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=80)
        return buf.getvalue(), int(im.width), int(im.height), comps
    except Exception:
        return None


@dataclass
class _Image:
    obj_index: int
    data: bytes
    width: int
    height: int
    components: int


@dataclass
class _Page:
    ops: List[bytes] = field(default_factory=list)
    # /ImN name -> index into PDFReport._images, for this page's resources.
    image_names: Dict[str, int] = field(default_factory=dict)
    # Pending link annotations: (x0, y0, x1, y1, uri)
    links: List[Tuple[float, float, float, float, str]] = field(default_factory=list)


# Brand palette shared by the layout helpers — 1-866 NOENTRY colors.
# The logo is a yellow warning-diamond with a black border + white "stop" hand
# and a "1-866 NOENTRY" wordmark (NO in red). We lead with black ink and red as
# the accent and use yellow sparingly (the emblem only), per brand guidance.
BRAND_YELLOW: "Color" = (0.965, 0.773, 0.0)  # #F6C500 warning yellow
BRAND_RED: "Color" = (0.851, 0.063, 0.051)   # #D9100D
NAVY: "Color" = (0.090, 0.094, 0.102)        # near-black brand ink (section bands)
ACCENT: "Color" = BRAND_RED                  # accent (titles, rules)
MUTED: "Color" = (0.42, 0.47, 0.55)
INK: "Color" = (0.10, 0.12, 0.16)
HAIRLINE: "Color" = (0.88, 0.90, 0.93)

# The official logo, rasterized from Frontend/rtsp-ui/public/logo-mark.svg. It is
# stored as JPEG because raw JPEG is the one format this writer embeds without
# Pillow (which is not a backend dependency); regenerate it with
# scripts/rasterize_logo.py if the SVG changes. ``brand_header`` falls back to
# the drawn emblem when the asset is missing, so reports never fail over a logo.
LOGO_PATH = Path(__file__).resolve().parent / "assets" / "logo-mark.jpg"
_logo_cache: Optional[Tuple[bytes, int, int, int]] = None
_logo_loaded = False


def brand_logo() -> Optional[Tuple[bytes, int, int, int]]:
    """The embeddable logo image, or None when the asset is unavailable."""
    global _logo_cache, _logo_loaded
    if not _logo_loaded:
        _logo_loaded = True
        try:
            _logo_cache = normalize_to_jpeg(LOGO_PATH.read_bytes())
        except Exception:
            _logo_cache = None
    return _logo_cache


class PDFReport:
    """Top-down flowing PDF document.

    The cursor ``self.y`` is the distance from the top margin; helpers append
    content and advance it, opening a new page automatically when space runs
    out. Call :meth:`render` to obtain the finished PDF as bytes.
    """

    def __init__(self) -> None:
        self._pages: List[_Page] = []
        self._images: List[_Image] = []
        self._cur: _Page = _Page()
        self._pages.append(self._cur)
        self.y = PAGE_H - MARGIN

    def _register_image(self, jpeg: Tuple[bytes, int, int, int]) -> str:
        """Add an image XObject and bind it to the current page's resources."""
        data, w, h, comps = jpeg
        img = _Image(obj_index=len(self._images), data=data, width=w, height=h, components=comps)
        self._images.append(img)
        name = "Im%d" % img.obj_index
        self._cur.image_names[name] = img.obj_index
        return name

    @property
    def content_width(self) -> float:
        return PAGE_W - 2 * MARGIN

    # -- page management ---------------------------------------------------
    def new_page(self) -> None:
        self._cur = _Page()
        self._pages.append(self._cur)
        self.y = PAGE_H - MARGIN

    def _ensure(self, needed: float) -> None:
        if self.y - needed < MARGIN:
            self.new_page()

    def spacer(self, height: float = 8.0) -> None:
        self.y -= height
        if self.y < MARGIN:
            self.new_page()

    # -- primitives --------------------------------------------------------
    def _draw_line_op(
        self, s: str, x: float, baseline: float, size: float, bold: bool, color: Color
    ) -> None:
        font = "F2" if bold else "F1"
        r, g, b = color
        esc = _escape_pdf_text(s)
        op = (
            b"BT /" + font.encode("ascii") + b" %.2f Tf %.3f %.3f %.3f rg "
            b"1 0 0 1 %.2f %.2f Tm (" % (size, r, g, b, x, baseline)
            + esc
            + b") Tj ET\n"
        )
        self._cur.ops.append(op)

    def _wrap(self, s: str, size: float, bold: bool, max_width: float) -> List[str]:
        lines: List[str] = []
        for raw_line in str(s).replace("\r\n", "\n").replace("\r", "\n").split("\n"):
            words = raw_line.split(" ")
            cur = ""
            for word in words:
                candidate = word if not cur else cur + " " + word
                if text_width(candidate, size, bold) <= max_width or not cur:
                    # Hard-break a single word that is wider than the column.
                    if not cur and text_width(word, size, bold) > max_width:
                        chunk = ""
                        for ch in word:
                            if text_width(chunk + ch, size, bold) <= max_width or not chunk:
                                chunk += ch
                            else:
                                lines.append(chunk)
                                chunk = ch
                        cur = chunk
                    else:
                        cur = candidate
                else:
                    lines.append(cur)
                    cur = word
            lines.append(cur)
        return lines

    def text(
        self,
        s: str,
        *,
        size: float = 11.0,
        bold: bool = False,
        color: Color = (0.1, 0.12, 0.16),
        indent: float = 0.0,
        leading: Optional[float] = None,
        space_after: float = 4.0,
    ) -> None:
        lead = leading if leading is not None else size * 1.32
        x = MARGIN + indent
        max_width = self.content_width - indent
        for line in self._wrap(s, size, bold, max_width):
            self._ensure(lead)
            baseline = self.y - size
            self._draw_line_op(line, x, baseline, size, bold, color)
            self.y -= lead
        self.y -= space_after

    def heading(self, s: str, *, size: float = 16.0, color: Color = (0.05, 0.07, 0.13)) -> None:
        self.text(s, size=size, bold=True, color=color, space_after=6.0)

    def label_value(self, label: str, value: str, *, size: float = 10.5) -> None:
        """One compact ``Label: value`` row, label in bold."""
        lead = size * 1.35
        self._ensure(lead)
        baseline = self.y - size
        label_txt = f"{label}: "
        self._draw_line_op(label_txt, MARGIN, baseline, size, True, (0.30, 0.36, 0.44))
        vx = MARGIN + text_width(label_txt, size, True)
        # Wrap the value within the remaining width; subsequent lines hang-indent.
        max_width = (PAGE_W - MARGIN) - vx
        value_lines = self._wrap(value, size, False, max_width)
        for idx, line in enumerate(value_lines):
            if idx > 0:
                self._ensure(lead)
                baseline = self.y - size
            self._draw_line_op(line, vx, baseline, size, False, (0.1, 0.12, 0.16))
            self.y -= lead
        self.y -= 2.0

    def hr(self, *, color: Color = (0.85, 0.87, 0.91)) -> None:
        self._ensure(10.0)
        y = self.y - 4
        r, g, b = color
        self._cur.ops.append(
            b"%.3f %.3f %.3f RG 0.7 w %.2f %.2f m %.2f %.2f l S\n"
            % (r, g, b, MARGIN, y, PAGE_W - MARGIN, y)
        )
        self.y -= 12.0

    def link(self, text: str, url: str, *, size: float = 10.5, indent: float = 0.0) -> None:
        """Draw a single-line clickable URL in link-blue with an underline."""
        url = str(url or "").strip()
        if not url:
            return
        color: Color = (0.07, 0.40, 0.85)
        # Truncate the visible text to one column width (the full URL still
        # opens via the annotation action).
        max_width = self.content_width - indent
        shown = text
        while text_width(shown, size, False) > max_width and len(shown) > 4:
            shown = shown[:-2]
        if shown != text:
            shown = shown[:-1] + "…" if shown else text
        lead = size * 1.4
        self._ensure(lead)
        baseline = self.y - size
        x = MARGIN + indent
        w = text_width(shown, size, False)
        self._draw_line_op(shown, x, baseline, size, False, color)
        # Underline.
        r, g, b = color
        self._cur.ops.append(
            b"%.3f %.3f %.3f RG 0.6 w %.2f %.2f m %.2f %.2f l S\n"
            % (r, g, b, x, baseline - 1.5, x + w, baseline - 1.5)
        )
        self._cur.links.append((x, baseline - 3, x + w, baseline + size, url))
        self.y -= lead + 2.0

    # -- branded layout helpers ---------------------------------------------
    def _fill_round_rect(self, x: float, y: float, w: float, h: float, r: float, color: Color) -> None:
        """Fill a rounded rectangle (used to compose the logo hand)."""
        r = min(r, w / 2.0, h / 2.0)
        k = 0.5523  # circle-to-bezier constant
        rr, gg, bb = color
        p = [
            b"%.2f %.2f m" % (x + r, y),
            b"%.2f %.2f l" % (x + w - r, y),
            b"%.2f %.2f %.2f %.2f %.2f %.2f c" % (x + w - r + r * k, y, x + w, y + r - r * k, x + w, y + r),
            b"%.2f %.2f l" % (x + w, y + h - r),
            b"%.2f %.2f %.2f %.2f %.2f %.2f c" % (x + w, y + h - r + r * k, x + w - r + r * k, y + h, x + w - r, y + h),
            b"%.2f %.2f l" % (x + r, y + h),
            b"%.2f %.2f %.2f %.2f %.2f %.2f c" % (x + r - r * k, y + h, x, y + h - r + r * k, x, y + h - r),
            b"%.2f %.2f l" % (x, y + r),
            b"%.2f %.2f %.2f %.2f %.2f %.2f c" % (x, y + r - r * k, x + r - r * k, y, x + r, y),
        ]
        self._cur.ops.append(b"%.3f %.3f %.3f rg " % (rr, gg, bb) + b" ".join(p) + b" f\n")

    def _draw_image_at(self, jpeg: Tuple[bytes, int, int, int], x: float, y: float, w: float, h: float) -> None:
        """Draw an image at an absolute position (x, y = lower-left, in points)."""
        name = self._register_image(jpeg)
        self._cur.ops.append(
            b"q %.2f 0 0 %.2f %.2f %.2f cm /%s Do Q\n" % (w, h, x, y, name.encode("ascii"))
        )

    def _draw_logo(self, cx: float, cy: float, box: float) -> None:
        """Draw the logo centered in a ``box``-sized square at (cx, cy).

        Uses the real logo asset when present, else the drawn emblem.
        """
        logo = brand_logo()
        if logo is None:
            self._draw_noentry_emblem(cx, cy, box)
            return
        _, iw, ih = logo[0], logo[1], logo[2]
        scale = box / float(max(iw, ih))
        w, h = iw * scale, ih * scale
        self._draw_image_at(logo, cx - w / 2.0, cy - h / 2.0, w, h)

    def _draw_noentry_emblem(self, cx: float, cy: float, box: float) -> None:
        """Draw the 1-866 NOENTRY warning-sign logo: a yellow diamond with a
        black border and a white 'stop' hand, centered at (cx, cy).

        Fallback for when the logo asset cannot be loaded — see ``_draw_logo``.
        """
        d = box / 2.0 * 0.98  # half-diagonal
        diamond = b"%.2f %.2f m %.2f %.2f l %.2f %.2f l %.2f %.2f l h" % (
            cx, cy + d, cx + d, cy, cx, cy - d, cx - d, cy,
        )
        yr, yg, yb = BRAND_YELLOW
        self._cur.ops.append(b"%.3f %.3f %.3f rg " % (yr, yg, yb) + diamond + b" f\n")
        # Black border.
        self._cur.ops.append(
            b"0.05 0.05 0.06 RG %.2f w " % max(1.7, box * 0.055) + diamond + b" S\n"
        )
        # White hand: palm + thumb + four fingers.
        s = box
        white: Color = (1.0, 1.0, 1.0)
        self._fill_round_rect(cx - 0.19 * s, cy - 0.26 * s, 0.38 * s, 0.36 * s, 0.06 * s, white)  # palm
        self._fill_round_rect(cx - 0.30 * s, cy - 0.10 * s, 0.15 * s, 0.11 * s, 0.055 * s, white)  # thumb
        fw, gap, fh = 0.070 * s, 0.021 * s, 0.20 * s
        total = 4 * fw + 3 * gap
        fx = cx - total / 2.0
        fy = cy + 0.06 * s
        for i in range(4):
            self._fill_round_rect(fx + i * (fw + gap), fy, fw, fh, fw / 2.0, white)

    def _draw_wordmark(self, x: float, baseline: float, size: float = 16.5) -> float:
        """Draw '1-866 NOENTRY' with NO in brand red, the rest in ink."""
        segs = (("1-866 ", NAVY), ("NO", BRAND_RED), ("ENTRY", NAVY))
        cx = x
        for text_seg, color in segs:
            self._draw_line_op(text_seg, cx, baseline, size, True, color)
            cx += text_width(text_seg, size, True)
        return cx

    def brand_header(self, *, company: str = "1-866 NOENTRY", tagline: str = "AI SECURITY MONITORING", right_text: str = "") -> None:
        """Company logo mark + wordmark at the top of the page, with an
        optional right-aligned meta line (e.g. the generation timestamp)."""
        if right_text:
            w = text_width(right_text, 8.5, False)
            self._draw_line_op(right_text, PAGE_W - MARGIN - w, self.y - 8.5, 8.5, False, MUTED)

        box = 46.0
        top = self.y
        bottom = top - box
        self._draw_logo(MARGIN + box / 2.0, bottom + box / 2.0, box)

        wx = MARGIN + box + 14.0
        self._draw_wordmark(wx, top - 20.0, 16.5)
        self._draw_line_op(tagline, wx, top - 20.0 - 12.0, 8.0, False, MUTED)

        self.y = bottom - 8.0
        # Full-width rule under the header.
        r, g, b = HAIRLINE
        self._cur.ops.append(
            b"%.3f %.3f %.3f RG 1.0 w %.2f %.2f m %.2f %.2f l S\n"
            % (r, g, b, MARGIN, self.y, PAGE_W - MARGIN, self.y)
        )
        self.y -= 14.0

    def title_center(self, s: str, *, size: float = 15.0, color: Color = ACCENT) -> None:
        """A centered report title line."""
        lead = size * 1.4
        self._ensure(lead)
        w = text_width(s, size, True)
        self._draw_line_op(s, (PAGE_W - w) / 2, self.y - size, size, True, color)
        self.y -= lead + 4.0

    def section_band(self, label: str, *, fill: Color = NAVY, text_color: Color = (1, 1, 1), size: float = 10.0) -> None:
        """A filled full-width band with bold text — section separator."""
        h = size + 11.0
        self._ensure(h + 6.0)
        top = self.y
        bottom = top - h
        r, g, b = fill
        self._cur.ops.append(
            b"%.3f %.3f %.3f rg %.2f %.2f %.2f %.2f re f\n"
            % (r, g, b, MARGIN, bottom, self.content_width, h)
        )
        self._draw_line_op(label, MARGIN + 9.0, bottom + (h - size) / 2 + 1.5, size, True, text_color)
        self.y = bottom - 9.0

    def hairline(self, *, color: Color = HAIRLINE) -> None:
        self._ensure(6.0)
        y = self.y - 2.5
        r, g, b = color
        self._cur.ops.append(
            b"%.3f %.3f %.3f RG 0.5 w %.2f %.2f m %.2f %.2f l S\n"
            % (r, g, b, MARGIN, y, PAGE_W - MARGIN, y)
        )
        self.y -= 7.0

    def field_label(self, label: str) -> None:
        """Just the small uppercase field label (e.g. above an image block)."""
        label_size = 7.4
        self._ensure(label_size * 1.4 + 4.0)
        self._draw_line_op(str(label).upper(), MARGIN, self.y - label_size, label_size, True, MUTED)
        self.y -= label_size * 1.4 + 3.0

    def field_row(self, label: str, value: Optional[str], *, size: float = 9.5, link_url: Optional[str] = None) -> None:
        """One record field: small uppercase label, wrapped value, hairline.

        When ``link_url`` is given, the value renders as a clickable link.
        Mirrors the label/value rows of a security activity report.
        """
        text_value = str(value).strip() if value is not None else ""
        # An empty field reads better as an explicit statement than as a dash.
        empty = not text_value and not link_url
        if empty:
            text_value = "Not recorded"

        label_size = 7.4
        lead = size * 1.38
        # Keep the label and at least the first value line together.
        self._ensure(label_size * 1.4 + lead + 10.0)

        self._draw_line_op(str(label).upper(), MARGIN, self.y - label_size, label_size, True, MUTED)
        self.y -= label_size * 1.4 + 2.0

        if link_url:
            shown = text_value or link_url
            max_width = self.content_width
            while text_width(shown, size, False) > max_width and len(shown) > 4:
                shown = shown[:-2]
            baseline = self.y - size
            w = text_width(shown, size, False)
            link_color: Color = (0.07, 0.40, 0.85)
            self._draw_line_op(shown, MARGIN, baseline, size, False, link_color)
            r, g, b = link_color
            self._cur.ops.append(
                b"%.3f %.3f %.3f RG 0.5 w %.2f %.2f m %.2f %.2f l S\n"
                % (r, g, b, MARGIN, baseline - 1.2, MARGIN + w, baseline - 1.2)
            )
            self._cur.links.append((MARGIN, baseline - 2.5, MARGIN + w, baseline + size, link_url))
            self.y -= lead
        else:
            value_color = MUTED if empty else INK
            for line in self._wrap(text_value, size, False, self.content_width):
                self._ensure(lead)
                self._draw_line_op(line, MARGIN, self.y - size, size, False, value_color)
                self.y -= lead

        self.hairline()

    def image_block(self, jpeg: Tuple[bytes, int, int, int], *, max_height: float = 260.0, link_url: Optional[str] = None) -> None:
        """A large flowed image (the event photo), scaled to fit the column."""
        data, iw, ih, comps = jpeg
        if iw <= 0 or ih <= 0:
            return
        draw_w = self.content_width
        scale = draw_w / float(iw)
        draw_h = ih * scale
        if draw_h > max_height:
            draw_h = max_height
            draw_w = iw * (max_height / float(ih))
        if draw_h + 8.0 > (PAGE_H - 2 * MARGIN):
            draw_h = PAGE_H - 2 * MARGIN - 8.0
            draw_w = iw * (draw_h / float(ih))
        self._ensure(draw_h + 8.0)

        name = self._register_image(jpeg)
        bottom = self.y - draw_h
        self._cur.ops.append(
            b"q %.2f 0 0 %.2f %.2f %.2f cm /%s Do Q\n"
            % (draw_w, draw_h, MARGIN, bottom, name.encode("ascii"))
        )
        if link_url:
            self._cur.links.append((MARGIN, bottom, MARGIN + draw_w, bottom + draw_h, link_url))
        self.y -= draw_h + 8.0

    # -- serialization -----------------------------------------------------
    def render(self) -> bytes:
        objects: List[bytes] = []  # 1-indexed; objects[i] is object (i+1)

        def add(obj: bytes) -> int:
            objects.append(obj)
            return len(objects)  # object number

        # Reserve: 1 Catalog, 2 Pages (filled in at the end).
        catalog_num = add(b"")
        pages_num = add(b"")

        # Shared fonts.
        font_regular = add(
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
        )
        font_bold = add(
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>"
        )

        # Image XObjects (one per embedded thumbnail).
        image_obj_nums: Dict[int, int] = {}
        for img in self._images:
            if img.components == 1:
                cs = b"/DeviceGray"
            elif img.components == 4:
                cs = b"/DeviceCMYK"
            else:
                cs = b"/DeviceRGB"
            header = (
                b"<< /Type /XObject /Subtype /Image /Width %d /Height %d "
                b"/ColorSpace %s /BitsPerComponent 8 /Filter /DCTDecode /Length %d >>\n"
                % (img.width, img.height, cs, len(img.data))
            )
            image_obj_nums[img.obj_index] = add(header + b"stream\n" + img.data + b"\nendstream")

        page_obj_nums: List[int] = []
        total_pages = len(self._pages)
        for page_no, page in enumerate(self._pages, start=1):
            # Page-number footer, bottom right.
            footer = "Page %d of %d" % (page_no, total_pages)
            fw = text_width(footer, 8.0, False)
            r, g, b = MUTED
            page.ops.append(
                b"BT /F1 8.00 Tf %.3f %.3f %.3f rg 1 0 0 1 %.2f %.2f Tm ("
                % (r, g, b, PAGE_W - MARGIN - fw, 28.0)
                + _escape_pdf_text(footer)
                + b") Tj ET\n"
            )

            # Link annotations -> indirect objects.
            annot_refs: List[int] = []
            for (x0, y0, x1, y1, uri) in page.links:
                uri_bytes = _escape_pdf_text(uri)
                annot = (
                    b"<< /Type /Annot /Subtype /Link /Border [0 0 0] "
                    b"/Rect [%.2f %.2f %.2f %.2f] /A << /S /URI /URI (%s) >> >>"
                    % (x0, y0, x1, y1, uri_bytes)
                )
                annot_refs.append(add(annot))

            content = b"".join(page.ops)
            compressed = zlib.compress(content)
            stream_obj = (
                b"<< /Length %d /Filter /FlateDecode >>\nstream\n" % len(compressed)
                + compressed
                + b"\nendstream"
            )
            content_num = add(stream_obj)

            xobjects = b""
            if page.image_names:
                parts = [
                    b"/%s %d 0 R" % (name.encode("ascii"), image_obj_nums[idx])
                    for name, idx in page.image_names.items()
                ]
                xobjects = b" /XObject << " + b" ".join(parts) + b" >>"

            annots = b""
            if annot_refs:
                annots = b" /Annots [" + b" ".join(b"%d 0 R" % n for n in annot_refs) + b"]"

            page_obj = (
                b"<< /Type /Page /Parent %d 0 R /MediaBox [0 0 %.2f %.2f] "
                b"/Resources << /Font << /F1 %d 0 R /F2 %d 0 R >>%s >> "
                b"/Contents %d 0 R%s >>"
                % (
                    pages_num,
                    PAGE_W,
                    PAGE_H,
                    font_regular,
                    font_bold,
                    xobjects,
                    content_num,
                    annots,
                )
            )
            page_obj_nums.append(add(page_obj))

        kids = b" ".join(b"%d 0 R" % n for n in page_obj_nums)
        objects[pages_num - 1] = (
            b"<< /Type /Pages /Count %d /Kids [%s] >>" % (len(page_obj_nums), kids)
        )
        objects[catalog_num - 1] = b"<< /Type /Catalog /Pages %d 0 R >>" % pages_num

        # Assemble the file with a cross-reference table.
        out = io.BytesIO()
        out.write(b"%PDF-1.5\n%\xe2\xe3\xcf\xd3\n")
        offsets: List[int] = []
        for i, obj in enumerate(objects, start=1):
            offsets.append(out.tell())
            out.write(b"%d 0 obj\n" % i)
            out.write(obj)
            out.write(b"\nendobj\n")

        xref_pos = out.tell()
        count = len(objects) + 1
        out.write(b"xref\n")
        out.write(b"0 %d\n" % count)
        out.write(b"0000000000 65535 f \n")
        for off in offsets:
            out.write(b"%010d 00000 n \n" % off)
        out.write(b"trailer\n")
        out.write(b"<< /Size %d /Root %d 0 R >>\n" % (count, catalog_num))
        out.write(b"startxref\n%d\n%%%%EOF\n" % xref_pos)
        return out.getvalue()
