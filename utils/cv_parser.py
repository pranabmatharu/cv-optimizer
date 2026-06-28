bash

cat > /home/claude/cv-optimizer/utils/cv_parser.py << 'PYEOF'
"""
cv_parser.py — Position-aware PDF parsing for Canva-exported CVs.
Detects headings by x0 position, restores fused words, extracts contact
from font-size metadata, and segments each section into individual entries.
"""

import re
import io
from collections import defaultdict
from typing import Optional


# ---------------------------------------------------------------------------
# Known CV section headings (normalised: uppercase, no spaces/symbols)
# ---------------------------------------------------------------------------
_KNOWN_HEADINGS = {
    "EDUCATION",
    "CONFERENCEPAPER", "CONFERENCEPAPERS",
    "INTERNSHIP&PROJECTS", "INTERNSHIPANDPROJECTS", "INTERNSHIPS&PROJECTS",
    "INTERNSHIPS", "INTERNSHIPPROJECTS",
    "PROJECTS",
    "POSITIONSOFRESPONSIBILITY", "POSITIONOFRESPONSIBILITY",
    "SKILLSANDEXPERTISE", "SKILLS", "TECHNICALSKILLS",
    "EXTRACURRICULARACTIVITIES", "EXTRACURRICULAR",
    "ACHIEVEMENTS", "AWARDS", "AWARDSANDACHIEVEMENTS",
    "CERTIFICATIONS", "CERTIFICATES",
    "PUBLICATIONS", "RESEARCHPAPERS",
    "RESEARCH",
    "EXPERIENCE", "WORKEXPERIENCE", "PROFESSIONALEXPERIENCE",
    "SUMMARY", "PROFESSIONALSUMMARY", "OBJECTIVE", "PROFILE",
    "LANGUAGES", "VOLUNTEERWORK", "LEADERSHIP",
}

_HEADING_DISPLAY = {
    "EDUCATION":                    "Education",
    "CONFERENCEPAPER":              "Conference Paper",
    "CONFERENCEPAPERS":             "Conference Papers",
    "INTERNSHIP&PROJECTS":          "Internship & Projects",
    "INTERNSHIPANDPROJECTS":        "Internship & Projects",
    "INTERNSHIPS&PROJECTS":         "Internship & Projects",
    "INTERNSHIPPROJECTS":           "Internship & Projects",
    "INTERNSHIPS":                  "Internships",
    "PROJECTS":                     "Projects",
    "POSITIONSOFRESPONSIBILITY":    "Positions of Responsibility",
    "POSITIONOFRESPONSIBILITY":     "Positions of Responsibility",
    "SKILLSANDEXPERTISE":           "Skills and Expertise",
    "SKILLS":                       "Skills",
    "TECHNICALSKILLS":              "Technical Skills",
    "EXTRACURRICULARACTIVITIES":    "Extracurricular Activities",
    "EXTRACURRICULAR":              "Extracurricular Activities",
    "ACHIEVEMENTS":                 "Achievements",
    "AWARDS":                       "Awards",
    "AWARDSANDACHIEVEMENTS":        "Awards & Achievements",
    "CERTIFICATIONS":               "Certifications",
    "CERTIFICATES":                 "Certificates",
    "PUBLICATIONS":                 "Publications",
    "RESEARCHPAPERS":               "Research Papers",
    "RESEARCH":                     "Research",
    "EXPERIENCE":                   "Experience",
    "WORKEXPERIENCE":               "Work Experience",
    "PROFESSIONALEXPERIENCE":       "Professional Experience",
    "SUMMARY":                      "Summary",
    "PROFESSIONALSUMMARY":          "Professional Summary",
    "OBJECTIVE":                    "Objective",
    "PROFILE":                      "Profile",
    "LANGUAGES":                    "Languages",
    "VOLUNTEERWORK":                "Volunteer Work",
    "LEADERSHIP":                   "Leadership",
}


# ---------------------------------------------------------------------------
# Space restoration for Canva-fused text
# ---------------------------------------------------------------------------

def _restore_spaces(text: str) -> str:
    """
    Insert spaces into Canva-fused words.
    'MachineLearning' -> 'Machine Learning'
    'IITKharagpur' -> 'IIT Kharagpur'
    'Grade12' -> 'Grade 12'
    """
    # lowercase -> Uppercase boundary: 'machineLearning' -> 'machine Learning'
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    # digit -> UpperCase (non-acronym): '2028Dual' -> '2028 Dual'
    text = re.sub(r'(\d)([A-Z][a-z])', r'\1 \2', text)
    # letter -> digit: 'Grade12' -> 'Grade 12'
    text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
    # collapse extra spaces
    return re.sub(r' {2,}', ' ', text).strip()


# ---------------------------------------------------------------------------
# Line grouping from word positions
# ---------------------------------------------------------------------------

def _group_into_lines(words: list, y_tol: float = 4.0) -> list:
    """
    Group pdfplumber word dicts by vertical position.
    Returns list of line dicts sorted top-to-bottom.
    """
    buckets = defaultdict(list)
    for w in words:
        key = round(w['top'] / y_tol) * y_tol
        buckets[key].append(w)

    lines = []
    for top in sorted(buckets):
        ws = sorted(buckets[top], key=lambda w: w['x0'])
        lines.append({
            'top':    top,
            'x0':     ws[0]['x0'],
            'x1':     ws[-1].get('x1', ws[-1]['x0'] + 50),
            'text':   ' '.join(w['text'] for w in ws),
            'words':  ws,
        })
    return lines


# ---------------------------------------------------------------------------
# Heading detection
# ---------------------------------------------------------------------------

_HEADING_X0_MIN = 150   # headings start past this x (center-aligned)


def _classify_line(line: dict, page_width: float) -> Optional[str]:
    """Return normalised heading key if this line is a section heading, else None."""
    if line['x0'] < _HEADING_X0_MIN:
        return None
    raw = line['text'].strip()
    # Normalise: uppercase, remove spaces and & symbols
    key = re.sub(r'[\s&\-–—]', '', raw).upper()
    if key in _KNOWN_HEADINGS:
        return key
    # Try stripping all non-alpha
    alpha_key = re.sub(r'[^A-Z]', '', key)
    for known in _KNOWN_HEADINGS:
        if re.sub(r'[^A-Z]', '', known) == alpha_key:
            return known
    return None


# ---------------------------------------------------------------------------
# Contact extraction (font-size aware)
# ---------------------------------------------------------------------------

_EMAIL_RE    = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_PHONE_RE    = re.compile(r"\+?\d[\d\s\-\.\(\)]{7,15}\d")
_LINKEDIN_RE = re.compile(r"linkedin\.com/in/[\w\-]+", re.IGNORECASE)
_GITHUB_RE   = re.compile(r"github\.com/[\w\-]+",      re.IGNORECASE)


def extract_contact_info(text: str, name_hint: str = "") -> dict:
    """Extract contact fields from header text."""
    email    = _EMAIL_RE.search(text)
    phone    = _PHONE_RE.search(text)
    linkedin = _LINKEDIN_RE.search(text)
    github   = _GITHUB_RE.search(text)

    name = name_hint or ""
    if not name:
        for line in text.splitlines():
            line = line.strip()
            if (line and not _EMAIL_RE.search(line) and not _PHONE_RE.search(line)
                    and 1 <= len(line.split()) <= 5 and line[0].isupper()):
                name = line
                break

    return {
        "name":     name,
        "email":    email.group()    if email    else "",
        "phone":    phone.group()    if phone    else "",
        "linkedin": linkedin.group() if linkedin else "",
        "github":   github.group()   if github   else "",
    }


def _extract_name_from_words(words: list) -> str:
    """
    Find the candidate's name by looking for the largest bold font
    in the first ~20 words of the PDF (the header area).
    """
    if not words:
        return ""
    header_words = [w for w in words if w.get('top', 999) < 60]
    if not header_words:
        header_words = words[:20]

    # Find max font size
    sizes = [w.get('size', 0) for w in header_words]
    if not sizes:
        return ""
    max_size = max(sizes)
    # Collect words at that size (with tolerance)
    name_words = [w for w in header_words
                  if abs(w.get('size', 0) - max_size) < 0.5]
    name_words.sort(key=lambda w: w['x0'])
    raw_name = ' '.join(w['text'] for w in name_words)
    return _restore_spaces(raw_name)


# ---------------------------------------------------------------------------
# Entry parsing
# ---------------------------------------------------------------------------

_DATE_RE = re.compile(
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*['\u2018\u2019]?\s*\d{2,4}"
    r"|(?:19|20)\d{2}",
    re.IGNORECASE,
)
_DATE_RANGE_RE = re.compile(
    r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*['\u2018\u2019\u2010]?\s*\d{2,4}"
    r"(?:\s*[-–—\u2010\u2011\u2012]\s*"
    r"(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*['\u2018\u2019]?\s*\d{2,4}|Present))?"
    r"|(?:June?|July?)\s*['\u2018\u2019]?\s*\d{2,4}"
    r"(?:\s*[-–—]\s*(?:Apr(?:il)?|May|Jun|Jul)\s*['\u2018\u2019]?\s*\d{2,4})?",
    re.IGNORECASE,
)
_BULLET_CHARS = set('•●▪▸-*')


def _is_bullet(line: str) -> bool:
    return bool(line.strip()) and line.strip()[0] in _BULLET_CHARS


def _is_entry_title(line: str) -> bool:
    """Heuristic: line with | separators or ending with a date = entry title."""
    stripped = line.strip()
    if not stripped or _is_bullet(stripped):
        return False
    has_pipe = '|' in stripped
    has_date = bool(_DATE_RANGE_RE.search(stripped))
    return (has_pipe or has_date) and len(stripped) > 8


def _parse_entries(section_key: str, content: str) -> list:
    """Split section content into individual role/project entries."""
    if not content.strip():
        return []

    lines = [l for l in content.splitlines() if l.strip()]
    entries = []
    current = None

    def flush():
        if current and (current['bullets'] or current['title']):
            entries.append(current)

    for line in lines:
        stripped = line.strip()
        restored = _restore_spaces(stripped)

        if _is_entry_title(restored):
            flush()
            date_m = _DATE_RANGE_RE.search(restored)
            date_str = date_m.group().strip() if date_m else ''
            title = restored[:date_m.start()].strip().rstrip('|‐ ') if date_m else restored
            current = {
                'title':     title,
                'date':      date_str,
                'objective': '',
                'bullets':   [],
                'raw':       restored,
            }
        elif restored.lower().startswith('objective:') and current:
            current['objective'] = restored[len('objective:'):].strip()
        elif _is_bullet(stripped) and current:
            bullet_text = _restore_spaces(stripped.lstrip(''.join(_BULLET_CHARS) + ' ').strip())
            current['bullets'].append(bullet_text)
        elif current:
            # Continuation of previous bullet (wrapped line)
            if current['bullets']:
                current['bullets'][-1] += ' ' + _restore_spaces(stripped)
            else:
                current['raw'] += ' ' + _restore_spaces(stripped)

    flush()

    # If nothing detected, make the whole section one entry
    if not entries:
        bullets = [
            _restore_spaces(l.strip().lstrip(''.join(_BULLET_CHARS) + ' '))
            for l in lines if _is_bullet(l)
        ]
        entries = [{
            'title':     _HEADING_DISPLAY.get(section_key, section_key.title()),
            'date':      '',
            'objective': '',
            'bullets':   bullets,
            'raw':       _restore_spaces(content),
        }]

    return entries


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------

def parse_cv(file_bytes: bytes) -> dict:
    """
    Parse a CV PDF into structured sections with position-aware heading detection.

    Returns:
    {
        'raw': str,
        'sections': {
            KEY: {
                'display_name': str,
                'content': str,
                'entries': [ { title, date, objective, bullets, raw } ]
            }
        },
        'section_order': [ KEY, ... ],
        'contact': { name, email, phone, linkedin, github }
    }
    """
    try:
        import pdfplumber
    except ImportError:
        raw = _extract_text_fallback(file_bytes)
        return _fallback_parse(raw)

    all_tagged_lines = []   # { top, x0, text, is_heading, heading_key }
    all_words_flat   = []   # for name extraction

    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_width = page.width
                words = page.extract_words(extra_attrs=['fontname', 'size'])
                all_words_flat.extend(words)

                if not words:
                    text = page.extract_text() or ""
                    for l in text.splitlines():
                        all_tagged_lines.append({
                            'top': 0, 'x0': 0,
                            'text': _restore_spaces(l),
                            'is_heading': False, 'heading_key': None,
                        })
                    continue

                lines = _group_into_lines(words)
                for line in lines:
                    hkey = _classify_line(line, page_width)
                    restored = _restore_spaces(line['text'])
                    all_tagged_lines.append({
                        'top':         line['top'],
                        'x0':          line['x0'],
                        'text':        restored,
                        'is_heading':  hkey is not None,
                        'heading_key': hkey,
                    })

    except Exception as e:
        raw = _extract_text_fallback(file_bytes)
        return _fallback_parse(raw)

    # ----- Segment into sections -----
    sections: dict = {}
    section_order: list = []
    current_key = 'HEADER'
    current_lines: list = []

    def flush():
        if current_key not in sections:
            sections[current_key] = {
                'display_name': _HEADING_DISPLAY.get(current_key, current_key.title()),
                'content': '',
                'entries': [],
            }
        block = '\n'.join(l['text'] for l in current_lines).strip()
        if block:
            sections[current_key]['content'] += ('\n' + block if sections[current_key]['content'] else block)

    for line in all_tagged_lines:
        if line['is_heading']:
            flush()
            current_lines = []
            current_key = line['heading_key']
            if current_key not in section_order:
                section_order.append(current_key)
        else:
            current_lines.append(line)
    flush()

    # ----- Contact -----
    name_hint = _extract_name_from_words(all_words_flat)
    header_text = sections.get('HEADER', {}).get('content', '')
    # Build a combined text that includes the raw header area for contact search
    contact_text = header_text + '\n' + ' '.join(
        w['text'] for w in all_words_flat[:30]
    )
    contact = extract_contact_info(contact_text, name_hint=name_hint)

    # ----- Parse entries -----
    for key in section_order:
        sections[key]['entries'] = _parse_entries(key, sections[key]['content'])

    raw = '\n'.join(l['text'] for l in all_tagged_lines)

    return {
        'raw':           raw,
        'sections':      sections,
        'section_order': section_order,
        'contact':       contact,
    }


# ---------------------------------------------------------------------------
# Fallback helpers
# ---------------------------------------------------------------------------

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Public helper: extract plain text from PDF bytes."""
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            parts = []
            for page in pdf.pages:
                words = page.extract_words()
                if words:
                    lines = _group_into_lines(words)
                    parts.extend(_restore_spaces(l['text']) for l in lines)
                else:
                    t = page.extract_text()
                    if t:
                        parts.append(t)
            return '\n'.join(parts)
    except Exception:
        return _extract_text_fallback(file_bytes)


def _extract_text_fallback(file_bytes: bytes) -> str:
    try:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        return '\n'.join(p.extract_text() or '' for p in reader.pages)
    except Exception:
        return ""


def _fallback_parse(raw: str) -> dict:
    sections: dict = {}
    section_order: list = []
    current_key = 'HEADER'
    buf: list = []

    def flush():
        if current_key not in sections:
            sections[current_key] = {
                'display_name': _HEADING_DISPLAY.get(current_key, current_key.title()),
                'content': '', 'entries': [],
            }
        block = '\n'.join(buf).strip()
        if block:
            sections[current_key]['content'] += ('\n' + block if sections[current_key]['content'] else block)

    for line in raw.splitlines():
        key = re.sub(r'[\s&\-]', '', line.strip()).upper()
        if key in _KNOWN_HEADINGS:
            flush(); buf = []
            current_key = key
            if key not in section_order:
                section_order.append(key)
        else:
            buf.append(line)
    flush()

    contact = extract_contact_info(sections.get('HEADER', {}).get('content', raw[:400]))
    for k in section_order:
        sections[k]['entries'] = _parse_entries(k, sections[k]['content'])

    return {'raw': raw, 'sections': sections, 'section_order': section_order, 'contact': contact}
PYEOF
echo "Done"
Output

