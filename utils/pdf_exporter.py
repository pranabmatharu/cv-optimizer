"""
pdf_exporter.py — Generate a clean PDF CV using ReportLab
"""

import io
import re
from datetime import datetime

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.lib import colors
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, HRFlowable, ListFlowable, ListItem
    )
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
    _REPORTLAB_AVAILABLE = True
except ImportError:
    _REPORTLAB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
_DARK    = colors.HexColor("#1a1a2e")
_ACCENT  = colors.HexColor("#16213e")
_BLUE    = colors.HexColor("#0f3460")
_MID     = colors.HexColor("#374151")
_LIGHT   = colors.HexColor("#6b7280")
_RULE    = colors.HexColor("#d1d5db")
_WHITE   = colors.white


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------

def _build_styles():
    base = getSampleStyleSheet()

    styles = {}

    styles["name"] = ParagraphStyle(
        "CVName",
        parent=base["Normal"],
        fontSize=22,
        textColor=_DARK,
        fontName="Helvetica-Bold",
        spaceAfter=2,
        alignment=TA_LEFT,
    )
    styles["contact"] = ParagraphStyle(
        "CVContact",
        parent=base["Normal"],
        fontSize=9,
        textColor=_LIGHT,
        fontName="Helvetica",
        spaceAfter=8,
        alignment=TA_LEFT,
    )
    styles["section_heading"] = ParagraphStyle(
        "CVSection",
        parent=base["Normal"],
        fontSize=11,
        textColor=_BLUE,
        fontName="Helvetica-Bold",
        spaceBefore=12,
        spaceAfter=3,
        alignment=TA_LEFT,
    )
    styles["body"] = ParagraphStyle(
        "CVBody",
        parent=base["Normal"],
        fontSize=10,
        textColor=_MID,
        fontName="Helvetica",
        spaceAfter=4,
        leading=14,
        alignment=TA_JUSTIFY,
    )
    styles["bullet"] = ParagraphStyle(
        "CVBullet",
        parent=base["Normal"],
        fontSize=10,
        textColor=_MID,
        fontName="Helvetica",
        spaceAfter=3,
        leftIndent=12,
        leading=14,
    )
    styles["job_title"] = ParagraphStyle(
        "CVJobTitle",
        parent=base["Normal"],
        fontSize=10,
        textColor=_DARK,
        fontName="Helvetica-Bold",
        spaceAfter=1,
    )
    styles["company"] = ParagraphStyle(
        "CVCompany",
        parent=base["Normal"],
        fontSize=9,
        textColor=_LIGHT,
        fontName="Helvetica-Oblique",
        spaceAfter=4,
    )

    return styles


def _hr(styles):
    return HRFlowable(width="100%", thickness=0.5, color=_RULE, spaceAfter=4)


def _section_header(title: str, styles, elements: list):
    elements.append(Paragraph(title.upper(), styles["section_heading"]))
    elements.append(_hr(styles))


def _para(text: str, style, elements: list):
    # Escape XML special chars for ReportLab
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    elements.append(Paragraph(text, style))


def _bullet_para(text: str, styles, elements: list):
    text = text.lstrip("•●▪▸-* ").strip()
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    elements.append(Paragraph(f"• {text}", styles["bullet"]))


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def _render_contact(contact: dict, styles, elements: list):
    parts = []
    if contact.get("email"):    parts.append(contact["email"])
    if contact.get("phone"):    parts.append(contact["phone"])
    if contact.get("linkedin"): parts.append(contact["linkedin"])
    if contact.get("github"):   parts.append(contact["github"])
    if parts:
        elements.append(Paragraph("  |  ".join(parts), styles["contact"]))


def _render_text_section(text: str, styles, elements: list):
    """Render a section as paragraphs / bullets."""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped[0] in "•●▪▸-*":
            _bullet_para(stripped, styles, elements)
        else:
            _para(stripped, styles["body"], elements)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_pdf(
    sections: dict,
    contact: dict,
    rewrites: dict | None = None,
) -> bytes:
    """
    Build and return a CV PDF as bytes.

    sections: output of cv_parser.parse_cv()
    contact:  output of cv_parser.extract_contact_info()
    rewrites: optional { original_bullet: rewritten_bullet } mapping to apply
    """
    if not _REPORTLAB_AVAILABLE:
        raise ImportError(
            "reportlab is not installed. Add `reportlab` to requirements.txt."
        )

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )
    styles   = _build_styles()
    elements = []

    # ----- Header: Name + Contact -----
    name = contact.get("name", "").strip() or "Your Name"
    elements.append(Paragraph(name, styles["name"]))
    _render_contact(contact, styles, elements)
    elements.append(Spacer(1, 0.2 * cm))

    # ----- Section order -----
    section_order = [
        ("summary",        "Professional Summary"),
        ("experience",     "Experience"),
        ("education",      "Education"),
        ("skills",         "Skills"),
        ("projects",       "Projects"),
        ("certifications", "Certifications"),
        ("awards",         "Awards & Recognition"),
        ("publications",   "Publications"),
        ("languages",      "Languages"),
        ("interests",      "Interests"),
    ]

    for key, title in section_order:
        text = sections.get(key, "").strip()
        if not text:
            continue

        _section_header(title, styles, elements)

        # Apply rewrites if provided
        if rewrites:
            for original, rewritten in rewrites.items():
                text = text.replace(original, rewritten)

        _render_text_section(text, styles, elements)

    # ----- Footer -----
    elements.append(Spacer(1, 0.5 * cm))
    generated = datetime.now().strftime("%d %b %Y")
    elements.append(Paragraph(
        f"<font color='#9ca3af' size='8'>Generated by CV Optimizer · {generated}</font>",
        styles["body"],
    ))

    doc.build(elements)
    return buffer.getvalue()


def generate_pdf_from_text(
    full_text: str,
    contact: dict,
    filename_hint: str = "cv_optimized",
) -> bytes:
    """
    Simpler version: render raw CV text into a clean PDF without section parsing.
    Useful as fallback when section segmentation isn't needed.
    """
    if not _REPORTLAB_AVAILABLE:
        raise ImportError("reportlab is not installed.")

    from utils.cv_parser import parse_cv
    sections = parse_cv(full_text)
    return generate_pdf(sections, contact)
