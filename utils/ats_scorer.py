"""
ats_scorer.py — 100-point ATS compliance engine across 5 dimensions
"""

import re
from typing import Any


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _section_present(sections: dict, *keys: str) -> bool:
    for k in keys:
        if sections.get(k, "").strip():
            return True
    return False


def _count_bullets(text: str) -> int:
    count = 0
    for line in text.splitlines():
        s = line.strip()
        if s and s[0] in "•●▪▸-*":
            count += 1
    return count


def _has_metrics(text: str) -> bool:
    """Check for quantified achievements (numbers / % / $)."""
    return bool(re.search(r"\b\d+[%+]?\b|\$\d+|\d+[kKmMbB]\b", text))


def _has_dates(text: str) -> bool:
    """Basic date / year range detection."""
    return bool(re.search(
        r"\b(19|20)\d{2}\b"
        r"|\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s*(19|20)?\d{2}\b"
        r"|\bPresent\b",
        text, re.IGNORECASE,
    ))


_FILLER_WORDS = [
    "responsible for", "duties included", "worked on", "helped with",
    "assisted in", "involved in", "participated in", "was part of",
    "tasked with",
]

_PASSIVE_PATTERNS = [
    r"\bwas\b", r"\bwere\b", r"\bbeen\b", r"\bbeing\b",
]

_POWER_VERBS = [
    "achieved", "built", "created", "delivered", "designed", "developed",
    "drove", "engineered", "established", "executed", "generated", "grew",
    "implemented", "improved", "increased", "launched", "led", "managed",
    "optimised", "optimized", "orchestrated", "reduced", "spearheaded",
    "streamlined", "transformed",
]

_ATS_UNFRIENDLY = [
    "table", "text box", "header", "footer", "image", "graph", "chart",
    "infographic", "column layout",
]


# ---------------------------------------------------------------------------
# Dimension scorers (each returns a dict with score, max, and findings list)
# ---------------------------------------------------------------------------

def _score_structure(sections: dict, raw: str) -> dict:
    """Dimension 1 — Document structure (20 pts)"""
    score = 0
    findings = []

    essential = {
        "summary":      ("Summary / Objective section", 4),
        "experience":   ("Work Experience section",     5),
        "education":    ("Education section",           4),
        "skills":       ("Skills section",              4),
    }
    for key, (label, pts) in essential.items():
        if _section_present(sections, key):
            score += pts
        else:
            findings.append(f"Missing {label}")

    # Bonus for extra sections
    bonus_sections = ["projects", "certifications", "awards", "publications"]
    bonus_count = sum(1 for s in bonus_sections if _section_present(sections, s))
    if bonus_count >= 1:
        score += 2
    if bonus_count >= 2:
        score += 1

    return {"score": min(score, 20), "max": 20, "findings": findings}


def _score_contact(contact: dict) -> dict:
    """Dimension 2 — Contact completeness (10 pts)"""
    score = 0
    findings = []

    fields = {
        "email":    ("Email address",    4),
        "phone":    ("Phone number",     3),
        "linkedin": ("LinkedIn profile", 2),
        "github":   ("GitHub profile",  1),
    }
    for field, (label, pts) in fields.items():
        if contact.get(field, "").strip():
            score += pts
        else:
            findings.append(f"Missing {label}")

    return {"score": score, "max": 10, "findings": findings}


def _score_content_quality(sections: dict) -> dict:
    """Dimension 3 — Content quality (30 pts)"""
    score = 0
    findings = []

    exp_text = sections.get("experience", "")

    # Bullet usage
    bullets = _count_bullets(exp_text)
    if bullets >= 8:
        score += 8
    elif bullets >= 4:
        score += 5
        findings.append("Add more bullet points to experience (aim for ≥8)")
    elif bullets >= 1:
        score += 2
        findings.append("Too few bullet points — expand experience section")
    else:
        findings.append("No bullet points found in experience section")

    # Quantified achievements
    if _has_metrics(exp_text):
        score += 8
    else:
        findings.append("Add quantified achievements (numbers, %, $, growth metrics)")

    # Date ranges present
    if _has_dates(exp_text) or _has_dates(sections.get("education", "")):
        score += 5
    else:
        findings.append("Date ranges missing from experience or education")

    # Power verbs
    exp_lower = exp_text.lower()
    used_power = [v for v in _POWER_VERBS if v in exp_lower]
    if len(used_power) >= 6:
        score += 5
    elif len(used_power) >= 3:
        score += 3
        findings.append("Use more action / power verbs (e.g. led, built, drove)")
    else:
        findings.append("Very few action verbs — bullets should start with strong verbs")

    # Penalise filler / passive language
    filler_hits = sum(1 for f in _FILLER_WORDS if f in exp_lower)
    passive_hits = sum(len(re.findall(p, exp_lower)) for p in _PASSIVE_PATTERNS)
    if filler_hits == 0 and passive_hits < 3:
        score += 4
    elif filler_hits <= 1 and passive_hits < 6:
        score += 2
        findings.append("Replace filler phrases ('responsible for', 'helped with') with direct action verbs")
    else:
        findings.append("Heavy use of passive / filler language — rewrite bullets in active voice")

    return {"score": min(score, 30), "max": 30, "findings": findings}


def _score_formatting(raw: str, sections: dict) -> dict:
    """Dimension 4 — Formatting & readability (20 pts)"""
    score = 0
    findings = []

    words = raw.split()
    word_count = len(words)

    # Length check (600–1200 words ≈ 1–2 page CV)
    if 500 <= word_count <= 1400:
        score += 7
    elif 300 <= word_count < 500:
        score += 4
        findings.append(f"CV is short ({word_count} words) — consider expanding")
    elif word_count > 1400:
        score += 4
        findings.append(f"CV is long ({word_count} words) — consider trimming to 1–2 pages")
    else:
        findings.append(f"CV is very short ({word_count} words)")

    # Consistent date format
    date_formats = [
        len(re.findall(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s*(19|20)\d{2}", raw, re.I)),
        len(re.findall(r"\b(0?[1-9]|1[0-2])/((19|20)\d{2})", raw)),
        len(re.findall(r"\b(19|20)\d{2}", raw)),
    ]
    if sum(1 for d in date_formats if d > 0) <= 1:
        score += 5
    else:
        score += 2
        findings.append("Use a single consistent date format throughout (e.g. Jan 2022)")

    # No ATS-unfriendly elements mentioned
    raw_lower = raw.lower()
    unfriendly_found = [t for t in _ATS_UNFRIENDLY if t in raw_lower]
    if not unfriendly_found:
        score += 5
    else:
        findings.append("Potential ATS-unfriendly elements detected — avoid tables, text boxes, columns")

    # Skills section not a wall of text
    skills_text = sections.get("skills", "")
    if skills_text:
        skill_bullets = _count_bullets(skills_text)
        skills_words = len(skills_text.split())
        if skill_bullets >= 3 or skills_words < 80:
            score += 3
        else:
            score += 1
            findings.append("Skills section may be a wall of text — consider grouping into categories")

    return {"score": min(score, 20), "max": 20, "findings": findings}


def _score_keyword_density(raw: str, sections: dict) -> dict:
    """Dimension 5 — Keyword density & skills (20 pts)"""
    score = 0
    findings = []

    skills_text = sections.get("skills", "")
    skills_lower = skills_text.lower()

    # Skills section has substance
    if len(skills_text.split()) >= 15:
        score += 6
    elif len(skills_text.split()) >= 5:
        score += 3
        findings.append("Expand skills section with more specific technologies / competencies")
    else:
        findings.append("Skills section is missing or nearly empty")

    # Technical keyword presence (generic check)
    tech_terms = [
        "python", "java", "javascript", "sql", "excel", "tableau", "power bi",
        "aws", "azure", "gcp", "docker", "kubernetes", "git", "agile", "scrum",
        "machine learning", "data analysis", "project management", "leadership",
    ]
    raw_lower = raw.lower()
    found_tech = [t for t in tech_terms if t in raw_lower]
    if len(found_tech) >= 8:
        score += 8
    elif len(found_tech) >= 4:
        score += 5
        findings.append("Include more specific technical keywords relevant to your field")
    elif len(found_tech) >= 1:
        score += 2
        findings.append("Very few recognisable keywords — ATS systems may rank you low")
    else:
        findings.append("No common technical keywords detected — add relevant skills")

    # Industry/role keywords in summary
    summary_text = sections.get("summary", "")
    if len(summary_text.split()) >= 30:
        score += 6
    elif len(summary_text.split()) >= 10:
        score += 3
        findings.append("Summary is short — expand it with key role-relevant keywords")
    else:
        findings.append("Add a strong professional summary packed with relevant keywords")

    return {"score": min(score, 20), "max": 20, "findings": findings}


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def calculate_ats_score(sections: dict, contact: dict) -> dict:
    """
    Run all 5 dimensions and return a consolidated result dict:
    {
        'total': int (0-100),
        'dimensions': { name: { score, max, findings } },
        'all_findings': [ str ],
        'grade': str,
    }
    """
    raw = sections.get("raw", "")

    d1 = _score_structure(sections, raw)
    d2 = _score_contact(contact)
    d3 = _score_content_quality(sections)
    d4 = _score_formatting(raw, sections)
    d5 = _score_keyword_density(raw, sections)

    dimensions = {
        "Structure":       d1,
        "Contact Info":    d2,
        "Content Quality": d3,
        "Formatting":      d4,
        "Keywords":        d5,
    }

    total = sum(d["score"] for d in dimensions.values())
    all_findings = []
    for d in dimensions.values():
        all_findings.extend(d["findings"])

    if total >= 85:
        grade = "Excellent"
    elif total >= 70:
        grade = "Good"
    elif total >= 55:
        grade = "Fair"
    elif total >= 40:
        grade = "Needs Work"
    else:
        grade = "Poor"

    return {
        "total":      total,
        "dimensions": dimensions,
        "all_findings": all_findings,
        "grade":      grade,
    }
