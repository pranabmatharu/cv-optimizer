"""
jd_matcher.py — Job Description semantic matching
Uses TF-IDF cosine similarity (no heavy ML deps, works on Streamlit Cloud)
"""
import re
import math
from collections import Counter
from dataclasses import dataclass


@dataclass
class SectionMatch:
    section: str
    score: float          # 0.0 - 1.0
    matched_terms: list[str]
    missing_terms: list[str]


@dataclass
class JDMatchResult:
    overall_score: float
    section_scores: list[SectionMatch]
    top_missing_skills: list[str]
    top_matched_skills: list[str]
    required_skills: list[str]
    nice_to_have_skills: list[str]
    keyword_heatmap: dict[str, float]   # term -> relevance score


STOP_WORDS = {
    "the", "and", "for", "with", "that", "this", "you", "are", "our",
    "will", "have", "from", "your", "about", "which", "their", "been",
    "they", "can", "also", "must", "who", "what", "when", "where", "how",
    "all", "but", "not", "any", "may", "its", "one", "two", "etc",
    "including", "such", "able", "well", "work", "team", "working",
    "experience", "skills", "role", "job", "position", "candidate",
    "please", "apply", "opportunity", "company", "join", "looking",
}

# Tech skills we want to specifically flag
TECH_SKILL_PATTERNS = [
    r"\b(python|java|javascript|typescript|golang|rust|c\+\+|kotlin|swift)\b",
    r"\b(react|vue|angular|next\.?js|node\.?js|express|django|flask|fastapi)\b",
    r"\b(aws|gcp|azure|docker|kubernetes|terraform|ansible|jenkins|ci/?cd)\b",
    r"\b(sql|postgresql|mysql|mongodb|redis|elasticsearch|kafka|rabbitmq)\b",
    r"\b(machine\s+learning|deep\s+learning|nlp|llm|pytorch|tensorflow|scikit.learn)\b",
    r"\b(rest(?:ful)?|graphql|grpc|api|microservices|serverless)\b",
    r"\b(git|agile|scrum|devops|mlops|data\s+science|analytics)\b",
]


def preprocess(text: str) -> list[str]:
    """Tokenize and clean text."""
    text = text.lower()
    text = re.sub(r"[^\w\s\+#]", " ", text)
    tokens = text.split()
    return [t for t in tokens if t not in STOP_WORDS and len(t) > 2]


def build_tfidf(docs: list[str]) -> list[dict[str, float]]:
    """Build TF-IDF vectors for a list of documents."""
    tokenized = [preprocess(d) for d in docs]
    n = len(tokenized)

    # Document frequency
    df: Counter = Counter()
    for tokens in tokenized:
        df.update(set(tokens))

    # TF-IDF for each doc
    vectors = []
    for tokens in tokenized:
        tf = Counter(tokens)
        total = len(tokens) or 1
        vec: dict[str, float] = {}
        for term, count in tf.items():
            tf_score = count / total
            idf_score = math.log((n + 1) / (df[term] + 1)) + 1
            vec[term] = tf_score * idf_score
        vectors.append(vec)

    return vectors


def cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    """Compute cosine similarity between two TF-IDF vectors."""
    shared = set(vec_a) & set(vec_b)
    dot = sum(vec_a[t] * vec_b[t] for t in shared)
    mag_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    mag_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def extract_tech_skills(text: str) -> list[str]:
    """Extract specific tech skills from text."""
    found = []
    text_lower = text.lower()
    for pattern in TECH_SKILL_PATTERNS:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        found.extend(matches)
    return list(set(found))


def extract_required_vs_nice(jd_text: str) -> tuple[list[str], list[str]]:
    """Try to split JD into required vs nice-to-have skills."""
    required = []
    nice = []

    # Split into paragraphs/sentences
    parts = re.split(r"\n+|(?<=[.!?])\s+", jd_text)

    nice_patterns = r"\b(nice\s+to\s+have|bonus|preferred|plus|advantageous|desirable|ideal)\b"
    in_nice_section = False

    for part in parts:
        if re.search(nice_patterns, part, re.IGNORECASE):
            in_nice_section = True
        if re.search(r"\b(required|must\s+have|essential|mandatory|minimum)\b", part, re.IGNORECASE):
            in_nice_section = False

        skills = extract_tech_skills(part)
        if in_nice_section:
            nice.extend(skills)
        else:
            required.extend(skills)

    # Fallback: all skills as required
    if not required and not nice:
        required = extract_tech_skills(jd_text)

    return list(set(required)), list(set(nice))


def match_cv_to_jd(cv_sections: dict, jd_text: str) -> JDMatchResult:
    """Match CV sections against job description."""
    if not jd_text.strip():
        return JDMatchResult(
            overall_score=0.0,
            section_scores=[],
            top_missing_skills=[],
            top_matched_skills=[],
            required_skills=[],
            nice_to_have_skills=[],
            keyword_heatmap={},
        )

    # Priority sections for matching
    priority_sections = ["summary", "experience", "skills", "projects"]
    section_weights = {
        "experience": 0.40,
        "skills": 0.30,
        "summary": 0.15,
        "projects": 0.10,
        "education": 0.03,
        "achievements": 0.02,
    }

    section_scores: list[SectionMatch] = []
    jd_tokens = set(preprocess(jd_text))
    jd_skills = extract_tech_skills(jd_text)
    required_skills, nice_skills = extract_required_vs_nice(jd_text)

    # Build TF-IDF for JD vs each section
    weighted_sum = 0.0
    weight_total = 0.0

    for sec_name, section in cv_sections.items():
        if not section.content.strip():
            continue

        docs = [jd_text, section.content]
        vectors = build_tfidf(docs)
        jd_vec, sec_vec = vectors[0], vectors[1]

        sim = cosine_similarity(jd_vec, sec_vec)

        # Which JD terms appear in this section?
        sec_tokens = set(preprocess(section.content))
        matched = [t for t in jd_tokens if t in sec_tokens and len(t) > 3]
        missing = [t for t in jd_tokens if t not in sec_tokens and len(t) > 4]

        section_scores.append(SectionMatch(
            section=sec_name,
            score=round(sim, 3),
            matched_terms=sorted(matched)[:15],
            missing_terms=sorted(missing)[:15],
        ))

        weight = section_weights.get(sec_name, 0.01)
        weighted_sum += sim * weight
        weight_total += weight

    overall = weighted_sum / weight_total if weight_total > 0 else 0.0
    overall = min(1.0, overall * 2.5)  # Scale up for better UX display

    # Full CV text for overall matching
    full_cv = " ".join(s.content for s in cv_sections.values())
    cv_skills = extract_tech_skills(full_cv)

    matched_skills = [s for s in jd_skills if s in full_cv.lower()]
    missing_skills = [s for s in jd_skills if s not in full_cv.lower()]

    # Build keyword heatmap — term -> how relevant it is
    jd_tokens_list = preprocess(jd_text)
    jd_freq = Counter(jd_tokens_list)
    total_jd = len(jd_tokens_list) or 1
    heatmap = {
        term: round((count / total_jd) * 100, 2)
        for term, count in jd_freq.most_common(40)
        if len(term) > 3
    }

    return JDMatchResult(
        overall_score=round(overall, 3),
        section_scores=sorted(section_scores, key=lambda x: x.score, reverse=True),
        top_missing_skills=missing_skills[:12],
        top_matched_skills=matched_skills[:12],
        required_skills=required_skills,
        nice_to_have_skills=nice_skills,
        keyword_heatmap=heatmap,
    )
