"""
gemini_client.py — All Gemini 2.5 Flash API calls
Handles bullet rewriting, section analysis, and overall CV feedback
"""
import streamlit as st
import google.generativeai as genai
from typing import Generator


def get_gemini_client():
    """Initialise Gemini client using Streamlit secrets."""
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    if not api_key:
        st.error("⚠️ Gemini API key not found. Add GEMINI_API_KEY to your Streamlit secrets.")
        st.stop()
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash")


def stream_gemini(prompt: str) -> Generator[str, None, None]:
    """Stream response from Gemini 2.5 Flash."""
    model = get_gemini_client()
    response = model.generate_content(prompt, stream=True)
    for chunk in response:
        if chunk.text:
            yield chunk.text


def rewrite_bullet(
    bullet: str,
    section: str,
    jd_context: str = "",
    tone: str = "professional",
) -> str:
    """Rewrite a single CV bullet using STAR format and strong verbs."""
    jd_hint = f"\nTarget job context: {jd_context[:300]}" if jd_context else ""

    prompt = f"""You are an expert CV/resume writer specialising in AI, ML, and software engineering roles.

Rewrite the following CV bullet point to:
1. Start with a strong action verb (Led, Built, Engineered, Designed, Deployed, Optimised, etc.)
2. Follow STAR format where possible (Situation/Task → Action → Result)
3. Add quantified impact if it can be reasonably inferred (%, time saved, scale)
4. Be concise (1-2 lines max)
5. Use tone: {tone}
6. Keep it truthful — don't invent specific numbers not implied by original{jd_hint}

Section: {section}
Original bullet: {bullet}

Return ONLY the rewritten bullet. No explanation, no preamble, no quotes."""

    model = get_gemini_client()
    response = model.generate_content(prompt)
    return response.text.strip().strip('"').strip("'")


def rewrite_section_bullets(
    section_name: str,
    bullets: list[str],
    jd_context: str = "",
) -> list[dict]:
    """Rewrite all bullets in a section, returning original + rewritten pairs."""
    results = []
    for bullet in bullets:
        if len(bullet.strip()) < 10:
            continue
        rewritten = rewrite_bullet(bullet, section_name, jd_context)
        results.append({
            "original": bullet,
            "rewritten": rewritten,
            "accepted": False,
            "skipped": False,
        })
    return results


def analyse_cv_overall(
    full_text: str,
    jd_text: str = "",
    ats_score: int = 0,
) -> str:
    """Generate a comprehensive CV analysis narrative."""
    jd_section = f"\n\nJob Description:\n{jd_text[:1500]}" if jd_text else ""

    prompt = f"""You are a senior technical recruiter and CV coach specialising in AI/ML/Software Engineering roles.

Analyse this CV and provide structured feedback. Be direct, specific, and actionable.

CV Content:
{full_text[:3000]}{jd_section}

ATS Score: {ats_score}/100

Provide feedback in this exact format:

## 💪 Strengths (3-4 points)
- [specific strength]

## ⚠️ Critical Improvements (3-5 points)  
- [specific, actionable improvement]

## 🎯 Quick Wins (2-3 things to fix today)
- [specific, fast change]

## 📊 Assessment
[2-3 sentence overall assessment. Be honest.]

Keep each point to 1-2 sentences. Focus on technical CV best practices."""

    model = get_gemini_client()
    response = model.generate_content(prompt)
    return response.text.strip()


def generate_summary(
    full_text: str,
    jd_text: str = "",
    existing_summary: str = "",
) -> str:
    """Generate or rewrite a professional summary section."""
    jd_hint = f"\nTailor it for this role:\n{jd_text[:800]}" if jd_text else ""
    existing = f"\nExisting summary to improve:\n{existing_summary}" if existing_summary else ""

    prompt = f"""You are an expert CV writer for AI/ML/Software Engineering roles.

Write a compelling 3-4 sentence professional summary for this CV.
- Lead with years of experience and specialisation
- Highlight 2-3 key technical strengths
- End with value proposition or career goal
- Do NOT use "I" or "My"
- Use third person or noun phrases

CV Content (for context):
{full_text[:2000]}{existing}{jd_hint}

Return ONLY the summary text. No labels, no quotes, no explanation."""

    model = get_gemini_client()
    response = model.generate_content(prompt)
    return response.text.strip()


def suggest_missing_skills(
    full_text: str,
    jd_text: str,
    missing_skills: list[str],
) -> str:
    """Suggest how to address skill gaps."""
    if not missing_skills:
        return "Your skills section appears well-aligned with the job description."

    prompt = f"""You are a technical career coach.

A candidate is applying for this role but their CV is missing these skills:
{', '.join(missing_skills[:10])}

Job Description excerpt:
{jd_text[:600]}

CV excerpt (skills/projects section):
{full_text[:1000]}

Give 3-4 specific, honest suggestions on how to address these gaps. 
Could they highlight adjacent skills? Reframe existing experience? Add a project?
Be realistic and encouraging.

Format as a short bullet list. No preamble."""

    model = get_gemini_client()
    response = model.generate_content(prompt)
    return response.text.strip()


def tailor_cv_for_jd(
    section_content: str,
    section_name: str,
    jd_text: str,
) -> str:
    """Rewrite an entire section to better target a specific JD."""
    prompt = f"""You are an expert CV writer. 

Rewrite this '{section_name}' section to better target the job description below.
- Emphasise relevant experience and skills
- Use keywords from the JD naturally
- Keep all facts truthful
- Maintain professional tone
- Preserve all real information — just reframe it

Job Description:
{jd_text[:800]}

Original {section_name} section:
{section_content[:1500]}

Return only the rewritten section content. No labels or explanation."""

    model = get_gemini_client()
    response = model.generate_content(prompt)
    return response.text.strip()
