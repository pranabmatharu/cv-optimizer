"""
jd_matcher.py — TF-IDF cosine similarity matching + skills gap detection
"""

import re
import math
from collections import Counter
from typing import Any


# ---------------------------------------------------------------------------
# Simple tokeniser + stop-word filter
# ---------------------------------------------------------------------------

_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "need",
    "that", "this", "these", "those", "it", "its", "we", "you", "he",
    "she", "they", "their", "our", "your", "my", "i", "me", "him", "her",
    "us", "them", "who", "which", "what", "when", "where", "how", "not",
    "no", "nor", "so", "yet", "both", "either", "neither", "each",
    "more", "most", "other", "some", "such", "than", "too", "very",
    "just", "also", "well", "into", "up", "out", "about", "over", "then",
    "if", "while", "although", "because", "since", "until", "unless",
    "through", "during", "before", "after", "above", "below", "between",
    "among", "within", "without", "across", "following", "including",
    "work", "experience", "responsibilities", "role", "position", "job",
    "candidate", "applicant", "looking", "seeking", "required", "preferred",
    "ability", "knowledge", "understanding", "strong", "excellent", "good",
    "relevant", "related", "including", "including", "eg", "ie",
}

_BIGRAM_SKILLS = [
    "machine learning", "deep learning", "natural language processing",
    "computer vision", "data science", "data engineering", "data analysis",
    "data visualization", "business intelligence", "project management",
    "product management", "software development", "software engineering",
    "web development", "mobile development", "full stack", "front end",
    "back end", "cloud computing", "devops", "site reliability",
    "artificial intelligence", "neural network", "large language model",
    "version control", "continuous integration", "continuous deployment",
    "agile methodology", "scrum framework", "test driven development",
    "object oriented", "functional programming", "microservices architecture",
    "restful api", "graphql api", "sql server", "postgresql database",
    "mongodb database", "redis cache", "apache kafka", "apache spark",
    "power bi", "google analytics", "tableau desktop",
    "microsoft excel", "google sheets",
]


def _tokenise(text: str) -> list[str]:
    """Lowercase, remove punctuation, split into tokens, remove stop words."""
    text = text.lower()
    # Preserve bigram skills before splitting
    for bigram in _BIGRAM_SKILLS:
        text = text.replace(bigram, bigram.replace(" ", "_"))
    tokens = re.findall(r"[a-z][a-z0-9_\-+#.]{1,}", text)
    return [t for t in tokens if t not in _STOP_WORDS and len(t) >= 3]


# ---------------------------------------------------------------------------
# TF-IDF helpers (two-document corpus: CV vs JD)
# ---------------------------------------------------------------------------

def _tf(tokens: list[str]) -> dict[str, float]:
    """Term frequency."""
    c = Counter(tokens)
    total = len(tokens) or 1
    return {term: count / total for term, count in c.items()}


def _idf(term: str, docs: list[list[str]]) -> float:
    """Inverse document frequency (with smoothing)."""
    df = sum(1 for doc in docs if term in doc)
    return math.log((len(docs) + 1) / (df + 1)) + 1


def _tfidf_vector(tokens: list[str], all_tokens_list: list[list[str]]) -> dict[str, float]:
    tf = _tf(tokens)
    return {
        term: score * _idf(term, all_tokens_list)
        for term, score in tf.items()
    }


def _cosine_similarity(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
    common = set(vec_a) & set(vec_b)
    if not common:
        return 0.0
    dot = sum(vec_a[t] * vec_b[t] for t in common)
    mag_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    mag_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# Skills extraction
# ---------------------------------------------------------------------------

# Hard-coded skill vocabulary (expand as needed)
_SKILL_VOCAB = {
    # Programming languages
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "rust",
    "ruby", "php", "swift", "kotlin", "r", "matlab", "scala", "perl",
    # Web / frameworks
    "react", "angular", "vue", "node.js", "django", "flask", "fastapi",
    "spring", "express", "nextjs", "html", "css", "sass",
    # Data / ML
    "pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras",
    "xgboost", "lightgbm", "spark", "hadoop", "airflow", "dbt",
    "machine_learning", "deep_learning", "natural_language_processing",
    "computer_vision", "data_science", "data_engineering", "data_analysis",
    "data_visualization", "nlp", "llm", "rag", "transformers",
    # Databases
    "sql", "mysql", "postgresql", "mongodb", "redis", "elasticsearch",
    "cassandra", "dynamodb", "neo4j", "sqlite",
    # Cloud / DevOps
    "aws", "azure", "gcp", "docker", "kubernetes", "terraform", "ansible",
    "jenkins", "github_actions", "circleci", "heroku", "vercel",
    "devops", "site_reliability", "continuous_integration",
    # BI / Analytics
    "tableau", "power_bi", "looker", "metabase", "google_analytics",
    "excel", "google_sheets", "qlik",
    # Tools / Practices
    "git", "jira", "confluence", "slack", "figma", "postman",
    "agile", "scrum", "kanban", "tdd", "ci/cd", "rest", "graphql",
    "microservices", "linux", "bash",
    # Soft / Generic
    "leadership", "communication", "project_management", "product_management",
    "stakeholder_management", "team_management", "presentation",
    "problem_solving", "analytical", "collaboration",
}


def _extract_skills(tokens: list[str]) -> set[str]:
    return {t for t in tokens if t in _SKILL_VOCAB}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def match_cv_to_jd(cv_text: str, jd_text: str) -> dict:
    """
    Compute similarity score + skills gap between CV and Job Description.

    Returns:
    {
        'similarity_score': float (0-100),
        'matched_skills': list[str],
        'missing_skills': list[str],
        'extra_skills': list[str],   # in CV but not in JD
        'top_jd_keywords': list[str],
        'keyword_coverage': float (0-100),
    }
    """
    cv_tokens  = _tokenise(cv_text)
    jd_tokens  = _tokenise(jd_text)

    docs = [cv_tokens, jd_tokens]
    cv_vec = _tfidf_vector(cv_tokens, docs)
    jd_vec = _tfidf_vector(jd_tokens, docs)

    similarity = _cosine_similarity(cv_vec, jd_vec)
    similarity_score = round(min(similarity * 100 * 2.5, 100), 1)  # scale to 0-100

    # Skills analysis
    cv_skills  = _extract_skills(cv_tokens)
    jd_skills  = _extract_skills(jd_tokens)

    matched_skills = sorted(cv_skills & jd_skills)
    missing_skills = sorted(jd_skills - cv_skills)
    extra_skills   = sorted(cv_skills - jd_skills)

    # Top JD keywords (by TF-IDF weight, excluding stop words & generic terms)
    jd_top = sorted(jd_vec.items(), key=lambda x: x[1], reverse=True)
    top_jd_keywords = [
        t.replace("_", " ") for t, _ in jd_top
        if t not in _STOP_WORDS and len(t) >= 4
    ][:20]

    # Keyword coverage: % of top JD keywords present in CV
    cv_token_set = set(cv_tokens)
    top_kws_raw = [t for t, _ in jd_top if t not in _STOP_WORDS and len(t) >= 4][:20]
    covered = sum(1 for kw in top_kws_raw if kw in cv_token_set)
    keyword_coverage = round((covered / len(top_kws_raw)) * 100, 1) if top_kws_raw else 0.0

    return {
        "similarity_score": similarity_score,
        "matched_skills":   [s.replace("_", " ") for s in matched_skills],
        "missing_skills":   [s.replace("_", " ") for s in missing_skills],
        "extra_skills":     [s.replace("_", " ") for s in extra_skills],
        "top_jd_keywords":  top_jd_keywords,
        "keyword_coverage": keyword_coverage,
    }


def get_missing_keywords(cv_text: str, jd_text: str, top_n: int = 15) -> list[str]:
    """Convenience: return just the top missing keywords to add to the CV."""
    result = match_cv_to_jd(cv_text, jd_text)
    # Combine missing skills + uncovered top keywords
    missing = set(result["missing_skills"])
    jd_tokens = _tokenise(jd_text)
    cv_tokens = _tokenise(cv_text)
    cv_set = set(cv_tokens)
    jd_vec = _tfidf_vector(jd_tokens, [cv_tokens, jd_tokens])
    for kw, _ in sorted(jd_vec.items(), key=lambda x: x[1], reverse=True):
        if kw not in cv_set and kw not in _STOP_WORDS and len(kw) >= 4:
            missing.add(kw.replace("_", " "))
        if len(missing) >= top_n:
            break
    return sorted(missing)[:top_n]
