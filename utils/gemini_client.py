"""
gemini_client.py — Gemini 2.5 Flash API wrapper for all AI-powered features
"""

import os
import json
import re
import streamlit as st

try:
    import google.generativeai as genai
    _GENAI_AVAILABLE = True
except ImportError:
    _GENAI_AVAILABLE = False


# ---------------------------------------------------------------------------
# Initialisation
# ---------------------------------------------------------------------------

def _get_api_key() -> str:
    """Retrieve Gemini API key from Streamlit secrets or environment."""
    # Streamlit Cloud secrets (preferred)
    try:
        key = st.secrets.get("GEMINI_API_KEY", "")
        if key:
            return key
    except Exception:
        pass
    # Environment variable fallback
    return os.environ.get("GEMINI_API_KEY", "")


def _get_model():
    """Return a configured Gemini model instance."""
    if not _GENAI_AVAILABLE:
        raise ImportError(
            "google-generativeai is not installed. "
            "Add `google-generativeai` to requirements.txt."
        )
    api_key = _get_api_key()
    if not api_key:
        raise ValueError(
            "Gemini API key not found. "
            "Add GEMINI_API_KEY to Streamlit secrets or environment variables."
        )
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash")  # fast + generous free tier


def _call(prompt: str, temperature: float = 0.4) -> str:
    """Low-level call. Returns the text response or raises."""
    model = _get_model()
    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=2048,
        ),
    )
    return response.text.strip()


def is_configured() -> bool:
    """Return True if an API key is available."""
    try:
        return bool(_get_api_key())
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Feature 1 — Bullet rewriter
# ---------------------------------------------------------------------------

def rewrite_bullet(bullet: str, job_description: str = "", tone: str = "professional") -> str:
    """
    Rewrite a single CV bullet point to be more impactful.
    tone: 'professional' | 'concise' | 'achievement-focused'
    """
    jd_context = (
        f"\n\nTarget job description (use relevant keywords):\n{job_description[:800]}"
        if job_description.strip()
        else ""
    )

    prompt = f"""You are an expert CV writer. Rewrite the following CV bullet point to be more impactful.

Guidelines:
- Start with a strong action verb (past tense)
- Include quantified results where possible (add placeholders like [X%] if no number given)
- Remove filler phrases ("responsible for", "helped with", "worked on")
- Keep to one sentence, under 25 words
- Tone: {tone}
- Return ONLY the rewritten bullet — no explanation, no prefix{jd_context}

Original bullet:
{bullet}

Rewritten bullet:"""

    return _call(prompt, temperature=0.35)


def rewrite_bullets_batch(bullets: list[str], job_description: str = "") -> list[dict]:
    """
    Rewrite multiple bullets at once.
    Returns list of { original, rewritten } dicts.
    """
    if not bullets:
        return []

    numbered = "\n".join(f"{i+1}. {b}" for i, b in enumerate(bullets))
    jd_context = (
        f"\n\nTarget job description:\n{job_description[:600]}"
        if job_description.strip()
        else ""
    )

    prompt = f"""You are an expert CV writer. Rewrite each bullet point below to be more impactful.

Rules:
- Start with a strong action verb (past tense)
- Quantify results where possible; use placeholders like [X%] if unknown
- Remove passive/filler language
- Max 25 words each
- Return ONLY a JSON array of strings — one rewritten bullet per item
- Preserve the original order{jd_context}

Bullets to rewrite:
{numbered}

Return format (strict JSON array, no markdown):
["rewritten bullet 1", "rewritten bullet 2", ...]"""

    raw = _call(prompt, temperature=0.35)
    # Strip markdown fences if present
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()

    try:
        rewritten = json.loads(raw)
        if isinstance(rewritten, list):
            results = []
            for i, orig in enumerate(bullets):
                results.append({
                    "original":  orig,
                    "rewritten": rewritten[i] if i < len(rewritten) else orig,
                })
            return results
    except (json.JSONDecodeError, IndexError):
        pass

    # Fallback: line-by-line
    lines = [l.strip().lstrip("0123456789.-) ") for l in raw.splitlines() if l.strip()]
    results = []
    for i, orig in enumerate(bullets):
        results.append({
            "original":  orig,
            "rewritten": lines[i] if i < len(lines) else orig,
        })
    return results


# ---------------------------------------------------------------------------
# Feature 2 — Professional summary generator
# ---------------------------------------------------------------------------

def generate_summary(cv_text: str, job_description: str = "", word_count: int = 80) -> str:
    """Generate a tailored professional summary."""
    jd_context = (
        f"\n\nTarget job description:\n{job_description[:800]}"
        if job_description.strip()
        else ""
    )

    prompt = f"""You are an expert CV writer. Write a professional summary for this candidate.

Requirements:
- Exactly {word_count}–{word_count + 20} words
- Open with years of experience + primary role/expertise
- Highlight 2–3 key strengths with specific skills
- End with value proposition / what they bring to the employer
- Packed with relevant keywords (for ATS)
- First-person free (no "I", "my")
- Return ONLY the summary text — no heading, no explanation{jd_context}

CV text:
{cv_text[:3000]}

Professional summary:"""

    return _call(prompt, temperature=0.4)


# ---------------------------------------------------------------------------
# Feature 3 — Skills gap analysis + recommendations
# ---------------------------------------------------------------------------

def analyse_skills_gap(cv_text: str, job_description: str, missing_skills: list[str]) -> dict:
    """
    AI-powered skills gap analysis returning structured recommendations.
    Returns { 'gap_analysis': str, 'quick_wins': list, 'learning_paths': list }
    """
    missing_str = ", ".join(missing_skills[:15]) if missing_skills else "none identified"

    prompt = f"""You are a career coach analysing a skills gap between a candidate's CV and a job description.

Missing skills identified: {missing_str}

Job description (excerpt):
{job_description[:1000]}

CV (excerpt):
{cv_text[:1500]}

Provide a JSON response with exactly these keys:
{{
  "gap_analysis": "2-3 sentence overall assessment",
  "quick_wins": ["skill or action that can be added to CV quickly", ...],  // 3-5 items
  "learning_paths": ["specific course/cert/resource to address a gap", ...],  // 3-5 items
  "transferable": ["existing CV skill that maps to a required skill", ...]  // 2-4 items
}}

Return ONLY valid JSON — no markdown fences, no explanation."""

    raw = _call(prompt, temperature=0.3)
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "gap_analysis": raw[:500] if raw else "Analysis unavailable.",
            "quick_wins": missing_skills[:5],
            "learning_paths": [],
            "transferable": [],
        }


# ---------------------------------------------------------------------------
# Feature 4 — Full CV review
# ---------------------------------------------------------------------------

def full_cv_review(cv_text: str, ats_score: int, jd_match: float) -> str:
    """Return a structured written review of the CV."""

    prompt = f"""You are a senior recruiter reviewing a CV. The CV has an ATS score of {ats_score}/100 and a {jd_match}% match to the target job.

Write a concise review covering:
1. **Overall Impression** (2 sentences)
2. **Top 3 Strengths**
3. **Top 3 Areas to Improve** (specific, actionable)
4. **Quick Wins** (2-3 things to fix in under 10 minutes)
5. **One-Line Verdict**

Use markdown formatting. Be direct and specific — avoid vague praise.

CV text:
{cv_text[:3000]}

Review:"""

    return _call(prompt, temperature=0.4)


# ---------------------------------------------------------------------------
# Feature 5 — Tailored cover letter
# ---------------------------------------------------------------------------

def generate_cover_letter(
    cv_text: str,
    job_description: str,
    company_name: str = "",
    role_title: str = "",
    tone: str = "professional",
) -> str:
    """Generate a tailored cover letter."""

    company_str = f" at {company_name}" if company_name else ""
    role_str    = f" for the {role_title} role" if role_title else ""

    prompt = f"""Write a compelling cover letter{role_str}{company_str}.

Tone: {tone}
Length: 3 paragraphs (~200 words total)
Structure:
- Para 1: Hook + why this role/company
- Para 2: 2 specific achievements from the CV that match JD requirements
- Para 3: Forward-looking close + call to action

Rules:
- Do NOT use "I am writing to express my interest" or "Please find attached"
- Match keywords from the job description
- Be specific, not generic
- Return ONLY the letter body (no subject line, no date, no address block)

Job description:
{job_description[:1000]}

CV highlights:
{cv_text[:2000]}

Cover letter:"""

    return _call(prompt, temperature=0.5)


# ---------------------------------------------------------------------------
# Feature 6 — Interview questions predictor
# ---------------------------------------------------------------------------

def predict_interview_questions(cv_text: str, job_description: str, n: int = 8) -> list[dict]:
    """
    Predict likely interview questions based on CV + JD.
    Returns list of { question, type, tip } dicts.
    """
    prompt = f"""You are an experienced interviewer. Based on this CV and job description, predict the {n} most likely interview questions.

For each question provide:
- question: the question text
- type: one of ["Behavioural", "Technical", "Situational", "Motivational"]
- tip: one-sentence tip on how to answer it well

Return ONLY a JSON array:
[{{"question": "...", "type": "...", "tip": "..."}}, ...]

Job description:
{job_description[:800]}

CV:
{cv_text[:1500]}

Questions JSON:"""

    raw = _call(prompt, temperature=0.4)
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()

    try:
        questions = json.loads(raw)
        if isinstance(questions, list):
            return questions[:n]
    except json.JSONDecodeError:
        pass

    # Fallback: extract lines that look like questions
    lines = [l.strip() for l in raw.splitlines() if "?" in l]
    return [{"question": l, "type": "General", "tip": ""} for l in lines[:n]]
