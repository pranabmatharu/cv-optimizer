"""
cv_parser.py — PDF extraction and section segmentation
"""
import re
import pdfplumber
import fitz
from dataclasses import dataclass, field
from typing import Optional

SECTION_PATTERNS = {
    "contact":     r"\b(contact|personal\s+info(?:rmation)?|profile|about\s+me)\b",
    "summary":     r"\b(summary|objective|professional\s+summary|career\s+objective|about)\b",
    "experience":  r"\b(experience|work\s+experience|employment|career\s+history|work\s+history)\b",
    "education":   r"\b(education|academic(?:s)?|qualifications?|degrees?)\b",
    "skills":      r"\b(skills?|technical\s+skills?|core\s+competencies|competencies|expertise)\b",
    "projects":    r"\b(projects?|personal\s+projects?|key\s+projects?|portfolio)\b",
    "certifications": r"\b(certifications?|certificates?|licen[sc]es?|credentials?)\b",
    "achievements": r"\b(achievements?|awards?|honours?|honors?|accomplishments?)\b",
    "languages":   r"\b(languages?|spoken\s+languages?)\b",
    "interests":   r"\b(interests?|hobbies|activities)\b",
    "references":  r"\b(references?|referees?)\b",
}

@dataclass
class CVSection:
    name: str
    raw_title: str
    content: str
    bullets: list = field(default_factory=list)
    line_start: int = 0
    line_end: int = 0

@dataclass
class ParsedCV:
    full_text: str
    sections: dict
    metadata: dict
    raw_lines: list
    page_count: int

def extract_text_from_pdf(file_bytes):
    text_pages = []
    page_count = 0
    try:
        import io
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            page_count = len(pdf.pages)
            for page in pdf.pages:
                page_text = page.extract_text(x_tolerance=3, y_tolerance=3)
                if page_text:
                    text_pages.append(page_text)
    except Exception:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        page_count = doc.page_count
        for page in doc:
            text_pages.append(page.get_text())
        doc.close()
    return "\n".join(text_pages), page_count

def extract_metadata(file_bytes):
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        meta = doc.metadata or {}
        doc.close()
        return {"author": meta.get("author", ""), "title": meta.get("title", ""), "creator": meta.get("creator", "")}
    except Exception:
        return {}

def detect_section(line):
    stripped = line.strip()
    if not stripped or len(stripped) > 60:
        return None
    is_header_like = (
        stripped.isupper() or stripped.istitle()
        or re.match(r"^[A-Z][A-Za-z\s&/]+$", stripped)
        or re.match(r"^[\W_]*[A-Z\s]+[\W_]*$", stripped)
    )
    if not is_header_like and len(stripped.split()) > 5:
        return None
    clean = stripped.lower().strip("•:|-_ ")
    for section_key, pattern in SECTION_PATTERNS.items():
        if re.search(pattern, clean, re.IGNORECASE):
            return section_key
    return None

def extract_bullets(text):
    bullets = []
    for line in text.split("\n"):
        s = line.strip()
        cleaned = re.sub(r"^[\u2022\u2023\u25e6\u2043\u2219\*\-\•\·]\s*", "", s)
        if cleaned and len(cleaned) > 15:
            bullets.append(cleaned)
    return bullets

def segment_sections(full_text, raw_lines):
    sections = {}
    current_section_key = None
    current_title = ""
    current_lines = []
    current_start = 0

    def flush(end_line):
        nonlocal current_section_key, current_title, current_lines, current_start
        if current_section_key and current_lines:
            content = "\n".join(current_lines).strip()
            sections[current_section_key] = CVSection(
                name=current_section_key, raw_title=current_title,
                content=content, bullets=extract_bullets(content),
                line_start=current_start, line_end=end_line,
            )
        current_lines = []

    for i, line in enumerate(raw_lines):
        detected = detect_section(line)
        if detected:
            flush(i)
            current_section_key = detected
            current_title = line.strip()
            current_start = i
        else:
            if current_section_key is not None:
                current_lines.append(line)
            elif line.strip():
                if "contact" not in sections:
                    sections["contact"] = CVSection(name="contact", raw_title="Contact", content="", line_start=0, line_end=i)
                sections["contact"].content += line + "\n"
    flush(len(raw_lines))
    return sections

def parse_cv(file_bytes):
    full_text, page_count = extract_text_from_pdf(file_bytes)
    metadata = extract_metadata(file_bytes)
    raw_lines = full_text.split("\n")
    sections = segment_sections(full_text, raw_lines)
    return ParsedCV(full_text=full_text, sections=sections, metadata=metadata, raw_lines=raw_lines, page_count=page_count)

def extract_contact_info(cv):
    text = cv.full_text
    email = re.findall(r"[\w.+-]+@[\w-]+\.[a-z]{2,}", text)
    phone = re.findall(r"[\+]?[\d\s\-().]{9,15}", text)
    linkedin = re.findall(r"linkedin\.com/in/[\w\-]+", text, re.IGNORECASE)
    github = re.findall(r"github\.com/[\w\-]+", text, re.IGNORECASE)
    return {
        "email": email[0] if email else "",
        "phone": phone[0].strip() if phone else "",
        "linkedin": linkedin[0] if linkedin else "",
        "github": github[0] if github else "",
    }
