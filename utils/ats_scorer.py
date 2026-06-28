"""
pdf_exporter.py — Generate a clean, ATS-friendly PDF from optimised CV content
Uses ReportLab for professional output
"""
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, HRFlowable, ListFlowable, ListItem
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER


# ── Colour palette ───────────────────────────────────────────────────────────
PRIMARY   = colors.HexColor("#6C63FF")
DARK      = colors.HexColor("#1A1A2E")
GREY      = colors.HexColor("#555566")
LIGHT     = colors.HexColor("#F0F0F8")
WHITE     = colors.white


def build_styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "name": ParagraphStyle(
            "name",
            fontName="Helvetica-Bold",
            fontSize=22,
            textColor=DARK,
            spaceAfter=4,
            alignment=TA_CENTER,
        ),
        "contact": ParagraphStyle(
            "contact",
            fontName="Helvetica",
            fontSize=9,
            textColor=GREY,
            spaceAfter=2,
            alignment=TA_CENTER,
        ),
        "section_title": ParagraphStyle(
            "section_title",
            fontName="Helvetica-Bold",
            fontSize=11,
            textColor=PRIMARY,
            spaceBefore=10,
            spaceAfter=2,
            textTransform="uppercase",
        ),
        "body": ParagraphStyle(
            "body",
            fontName="Helvetica",
            fontSize=9.5,
            textColor=DARK,
            spaceAfter=3,
            leading=14,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            fontName="Helvetica",
            fontSize=9.5,
            textColor=DARK,
            leftIndent=12,
            spaceAfter=2,
            leading=13,
        ),
        "label": ParagraphStyle(
            "label",
            fontName="Helvetica-Bold",
            fontSize=9.5,
            textColor=DARK,
            spaceAfter=0,
        ),
    }


SECTION_ORDER = [
    "contact", "summary", "experience", "education",
    "skills", "projects", "certifications", "achievements",
    "languages", "interests",
]


def export_cv_to_pdf(
    name: str,
    contact_info: dict,
    sections: dict,           # section_name -> content string
    accepted_rewrites: dict,  # section_name -> list of {original, rewritten}
) -> bytes:
    """
    Build a clean ATS-friendly PDF.
    accepted_rewrites: rewrites the user has accepted replace original bullets.
    """
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=1.8 * cm,
        rightMargin=1.8 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
    )

    styles = build_styles()
    story = []

    # ── Header ───────────────────────────────────────────────────────────────
    if name:
        story.append(Paragraph(name, styles["name"]))

    contact_parts = []
    for key in ("email", "phone", "linkedin", "github"):
        val = contact_info.get(key, "")
        if val:
            contact_parts.append(val)
    if contact_parts:
        story.append(Paragraph("  |  ".join(contact_parts), styles["contact"]))

    story.append(Spacer(1, 6))
    story.append(HRFlowable(width="100%", thickness=1.5, color=PRIMARY))
    story.append(Spacer(1, 6))

    # ── Sections (ordered) ───────────────────────────────────────────────────
    ordered_keys = [k for k in SECTION_ORDER if k in sections and k != "contact"]
    # Append any sections not in the order list
    ordered_keys += [k for k in sections if k not in ordered_keys and k != "contact"]

    for sec_name in ordered_keys:
        content = sections.get(sec_name, "")
        if not content.strip():
            continue

        # Section title
        story.append(Paragraph(sec_name.upper().replace("_", " "), styles["section_title"]))
        story.append(HRFlowable(width="100%", thickness=0.5, color=LIGHT))
        story.append(Spacer(1, 3))

        # Apply accepted rewrites for this section
        rewrites = accepted_rewrites.get(sec_name, [])
        rewrite_map = {
            r["original"]: r["rewritten"]
            for r in rewrites
            if r.get("accepted")
        }

        # Split content into lines and render
        lines = content.split("\n")
        for line in lines:
            stripped = line.strip()
            if not stripped:
                story.append(Spacer(1, 2))
                continue

            # Check if this line has an accepted rewrite
            display = rewrite_map.get(stripped, stripped)

            # Detect bullet
            is_bullet = bool(
                stripped.startswith(("•", "-", "–", "*", "·", "◦"))
                or (len(stripped) > 0 and stripped[0] in "•-–*·◦")
            )

            if is_bullet:
                clean = display.lstrip("•-–*·◦ ").strip()
                story.append(Paragraph(f"• {clean}", styles["bullet"]))
            else:
                story.append(Paragraph(display, styles["body"]))

        story.append(Spacer(1, 4))

    doc.build(story)
    return buf.getvalue()
