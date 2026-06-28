"""
cv_parser.py — PDF text extraction + section segmentation + contact info extraction
"""

import re
import io
from typing import Optional

# ---------------------------------------------------------------------------
# PDF text extraction (pdfplumber preferred; PyPDF2 as fallback)
# ---------------------------------------------------------------------------

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract raw text from a PDF given its bytes."""
    text = ""

    # Try pdfplumber first (better layout preservation)
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            pages = []
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    pages.append(page_text)
            text = "\n".join(pages)
        if text.strip():
            return text
    except Exception:
        pass

    # Fallback: PyPDF2
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        pages = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                pages.append(page_text)
        text = "\n".join(pages)
    except Exception:
        pass

    return text


# ---------------------------------------------------------------------------
# Contact info extraction
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(
    r"(?:\+?\d{1,3}[\s\-.]?)?"
    r"(?:\(?\d{2,4}\)?[\s\-.]?)"
    r"\d{3,4}[\s\-.]?\d{3,4}"
)
_LINKEDIN_RE = re.compile(r"linkedin\.com/in/[\w\-]+", re.IGNORECASE)
_GITHUB_RE   = re.compile(r"github\.com/[\w\-]+",   re.IGNORECASE)


def extract_contact_info(text: str) -> dict:
    """Return a dict of contact fields found in the CV text."""
    email    = _EMAIL_RE.search(text)
    phone    = _PHONE_RE.search(text)
    linkedin = _LINKEDIN_RE.search(text)
    github   = _GITHUB_RE.search(text)

    # Attempt to grab the name from the very first non-empty line
    name = ""
    for line in text.splitlines():
        line = line.strip()
        if line and not _EMAIL_RE.search(line) and not _PHONE_RE.search(line):
            # Rough heuristic: short line near the top likely to be a name
            if len(line.split()) <= 6:
                name = line
                break

    return {
        "name":     name,
        "email":    email.group()    if email    else "",
        "phone":    phone.group()    if phone    else "",
        "linkedin": linkedin.group() if linkedin else "",
        "github":   github.group()   if github   else "",
    }


# ---------------------------------------------------------------------------
# Section segmentation
# ---------------------------------------------------------------------------

# Ordered list of canonical section labels and their common aliases
_SECTION_ALIASES = {
    "summary": [
        "summary", "professional summary", "objective", "profile",
        "about me", "career objective", "personal statement",
    ],
    "experience": [
        "experience", "work experience", "employment", "professional experience",
        "career history", "work history", "positions held",
    ],
    "education": [
        "education", "academic background", "qualifications",
        "academic qualifications", "educational background",
    ],
    "skills": [
        "skills", "technical skills", "core competencies", "competencies",
        "key skills", "technologies", "tools", "expertise",
    ],
    "projects": [
        "projects", "personal projects", "key projects", "portfolio",
        "notable projects", "selected projects",
    ],
    "certifications": [
        "certifications", "certificates", "licenses", "accreditations",
        "professional development",
    ],
    "awards": [
        "awards", "honours", "honors", "achievements", "recognition",
    ],
    "publications": [
        "publications", "papers", "research", "articles",
    ],
    "languages": [
        "languages", "language skills",
    ],
    "interests": [
        "interests", "hobbies", "extracurricular",
    ],
    "references": [
        "references", "referees",
    ],
}

# Build a flat lookup: alias_lower -> canonical
_ALIAS_MAP: dict[str, str] = {}
for _canonical, _aliases in _SECTION_ALIASES.items():
    for _alias in _aliases:
        _ALIAS_MAP[_alias.lower()] = _canonical

# A heading line is typically short, title-cased or ALL-CAPS, optionally
# followed by a separator.  We use a broad pattern and then check the alias map.
_HEADING_RE = re.compile(
    r"^(?P<title>[A-Z][A-Za-z &/\-]{1,50})"   # heading text
    r"\s*[:\-–—]?\s*$",                         # optional colon/dash at end
    re.MULTILINE,
)


def _classify_heading(line: str) -> Optional[str]:
    """Return canonical section name if the line looks like a section heading."""
    clean = line.strip().rstrip(":–—-").strip()
    return _ALIAS_MAP.get(clean.lower())


def parse_cv(text: str) -> dict:
    """
    Segment the CV text into sections.
    Returns a dict: { canonical_section: text_content, ... }
    Also includes a 'raw' key with the full text.
    """
    sections: dict[str, str] = {"raw": text}

    lines = text.splitlines()
    current_section = "header"
    buffer: list[str] = []

    def flush():
        if buffer:
            content = "\n".join(buffer).strip()
            if content:
                sections.setdefault(current_section, "")
                sections[current_section] += ("\n" + content if sections[current_section] else content)

    for line in lines:
        canonical = _classify_heading(line)
        if canonical:
            flush()
            buffer = []
            current_section = canonical
        else:
            buffer.append(line)

    flush()  # last section

    return sections


# ---------------------------------------------------------------------------
# Quick summary stats
# ---------------------------------------------------------------------------

def word_count(text: str) -> int:
    return len(text.split())


def bullet_count(text: str) -> int:
    """Count lines that start with common bullet characters."""
    bullets = 0
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and stripped[0] in "•●▪▸-*":
            bullets += 1
    return bullets
