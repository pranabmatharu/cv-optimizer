"""
app.py — CV Optimizer · Streamlit Cloud entry point
"""

import io
import streamlit as st

# ── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="CV Optimizer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Local imports (utils package) ───────────────────────────────────────────
from utils.cv_parser import extract_text_from_pdf, parse_cv, extract_contact_info
from utils.ats_scorer import calculate_ats_score
from utils.jd_matcher import match_cv_to_jd
from utils.gemini_client import (
    is_configured,
    rewrite_bullet,
    rewrite_bullets_batch,
    generate_summary,
    analyse_skills_gap,
    full_cv_review,
    generate_cover_letter,
    predict_interview_questions,
)
from utils.pdf_exporter import generate_pdf


# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Global */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Metric cards */
div[data-testid="metric-container"] {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 12px 16px;
}

/* Score colour coding via data attrs */
.score-excellent { color: #059669; font-weight: 700; }
.score-good      { color: #2563eb; font-weight: 700; }
.score-fair      { color: #d97706; font-weight: 700; }
.score-poor      { color: #dc2626; font-weight: 700; }

/* Tab styling */
button[data-baseweb="tab"] { font-size: 0.9rem; font-weight: 500; }

/* Sidebar */
section[data-testid="stSidebar"] > div { padding-top: 1.5rem; }

/* Bullet cards */
.bullet-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 12px 16px;
    margin-bottom: 10px;
}
.bullet-original { color: #6b7280; font-size: 0.9rem; margin-bottom: 6px; }
.bullet-new      { color: #111827; font-size: 0.95rem; font-weight: 500; }

/* Finding badges */
.finding-badge {
    background: #fef3c7;
    border-left: 3px solid #f59e0b;
    padding: 6px 10px;
    border-radius: 0 6px 6px 0;
    margin: 4px 0;
    font-size: 0.88rem;
}
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ─────────────────────────────────────────────
def _init_state():
    defaults = {
        "cv_text":       "",
        "cv_sections":   {},
        "contact_info":  {},
        "ats_result":    {},
        "jd_match":      {},
        "jd_text":       "",
        "rewrites":      {},      # original -> rewritten
        "accepted":      set(),   # set of originals accepted by user
        "file_name":     "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ── Helpers ──────────────────────────────────────────────────────────────────
def _score_color(score: int) -> str:
    if score >= 85: return "#059669"
    if score >= 70: return "#2563eb"
    if score >= 55: return "#d97706"
    return "#dc2626"


def _score_emoji(score: int) -> str:
    if score >= 85: return "🟢"
    if score >= 70: return "🔵"
    if score >= 55: return "🟡"
    return "🔴"


def _extract_bullets(text: str) -> list[str]:
    """Extract bullet-point lines from a block of text."""
    bullets = []
    for line in text.splitlines():
        s = line.strip()
        if s and s[0] in "•●▪▸-*" and len(s) > 5:
            bullets.append(s.lstrip("•●▪▸-* ").strip())
    return bullets


def _run_analysis():
    """Run ATS scoring + JD matching and cache in session state."""
    if not st.session_state["cv_text"]:
        return
    sections = st.session_state["cv_sections"]
    contact  = st.session_state["contact_info"]
    st.session_state["ats_result"] = calculate_ats_score(sections, contact)
    if st.session_state["jd_text"]:
        st.session_state["jd_match"] = match_cv_to_jd(
            st.session_state["cv_text"],
            st.session_state["jd_text"],
        )


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 CV Optimizer")
    st.caption("Upload your CV to get started")
    st.divider()

    # API key status
    if is_configured():
        st.success("✅ Gemini API connected", icon="🤖")
    else:
        st.warning("⚠️ Gemini API key missing\n\nAdd `GEMINI_API_KEY` to Streamlit secrets to enable AI features.", icon="🔑")

    st.divider()

    # CV upload
    uploaded_file = st.file_uploader(
        "Upload your CV (PDF)",
        type=["pdf"],
        help="Upload a PDF version of your CV/resume",
    )

    if uploaded_file and uploaded_file.name != st.session_state["file_name"]:
        with st.spinner("Extracting CV text…"):
            file_bytes = uploaded_file.read()
            raw_text   = extract_text_from_pdf(file_bytes)

        if raw_text.strip():
            st.session_state["cv_text"]     = raw_text
            st.session_state["cv_sections"] = parse_cv(raw_text)
            st.session_state["contact_info"] = extract_contact_info(raw_text)
            st.session_state["file_name"]   = uploaded_file.name
            st.session_state["rewrites"]    = {}
            st.session_state["accepted"]    = set()
            _run_analysis()
            st.success(f"✅ Loaded: {uploaded_file.name}")
        else:
            st.error("Could not extract text from this PDF. Try a text-based PDF (not a scanned image).")

    st.divider()

    # Job description input
    st.markdown("**Job Description** *(optional but recommended)*")
    jd_input = st.text_area(
        "Paste the job description here",
        value=st.session_state["jd_text"],
        height=180,
        placeholder="Paste the full job description to get match score and tailored recommendations…",
        label_visibility="collapsed",
    )
    if jd_input != st.session_state["jd_text"]:
        st.session_state["jd_text"] = jd_input
        if st.session_state["cv_text"]:
            _run_analysis()

    st.divider()

    # CV stats
    if st.session_state["cv_text"]:
        words = len(st.session_state["cv_text"].split())
        bullets = sum(
            1 for l in st.session_state["cv_text"].splitlines()
            if l.strip() and l.strip()[0] in "•●▪▸-*"
        )
        sections_found = [
            k for k in st.session_state["cv_sections"]
            if k != "raw" and st.session_state["cv_sections"][k].strip()
        ]
        st.markdown("**CV stats**")
        col1, col2 = st.columns(2)
        col1.metric("Words", words)
        col2.metric("Bullets", bullets)
        st.caption(f"Sections: {', '.join(sections_found) or 'none detected'}")


# ── Main content ─────────────────────────────────────────────────────────────

if not st.session_state["cv_text"]:
    # Landing / empty state
    st.markdown("# 📄 CV Optimizer")
    st.markdown("*AI-powered CV analysis, ATS scoring, and bullet rewriting — powered by Gemini 2.5 Flash*")
    st.divider()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📊 ATS Score")
        st.write("Get a 100-point ATS compliance score across 5 dimensions: structure, contact info, content quality, formatting, and keywords.")
    with col2:
        st.markdown("### 🎯 JD Matching")
        st.write("Paste a job description to see your match %, missing skills, and the keywords you need to add.")
    with col3:
        st.markdown("### ✏️ AI Rewriter")
        st.write("Let Gemini rewrite your bullet points to be stronger, more impact-driven, and ATS-optimised.")

    st.divider()
    st.info("👈 Upload your CV PDF in the sidebar to get started", icon="⬅️")
    st.stop()


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 ATS Score",
    "🎯 JD Match",
    "✏️ Rewriter",
    "💬 AI Tools",
    "📥 Export",
])


# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — ATS Score
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("## 📊 ATS Compliance Score")

    ats = st.session_state.get("ats_result", {})
    if not ats:
        st.info("Run analysis by uploading your CV in the sidebar.")
        st.stop()

    total = ats["total"]
    grade = ats["grade"]
    color = _score_color(total)

    # Big score display
    st.markdown(
        f"<h1 style='font-size:3.5rem; color:{color}; margin:0'>"
        f"{total}<span style='font-size:1.5rem; color:#6b7280'>/100</span></h1>"
        f"<p style='color:{color}; font-size:1.1rem; margin-top:0'>{_score_emoji(total)} {grade}</p>",
        unsafe_allow_html=True,
    )

    st.divider()

    # Dimension breakdown
    st.markdown("### Dimension Breakdown")
    dims = ats.get("dimensions", {})
    cols = st.columns(len(dims))
    for col, (dim_name, dim_data) in zip(cols, dims.items()):
        pct = int(dim_data["score"] / dim_data["max"] * 100)
        col.metric(
            label=dim_name,
            value=f"{dim_data['score']}/{dim_data['max']}",
            delta=f"{pct}%",
        )

    st.divider()

    # Visual progress bars
    st.markdown("### Score by Dimension")
    for dim_name, dim_data in dims.items():
        pct = dim_data["score"] / dim_data["max"]
        bar_color = _score_color(int(pct * 100))
        st.markdown(f"**{dim_name}** — {dim_data['score']}/{dim_data['max']}")
        st.progress(pct)

    st.divider()

    # Findings / recommendations
    findings = ats.get("all_findings", [])
    if findings:
        st.markdown("### 🔧 Recommendations")
        for f in findings:
            st.markdown(
                f"<div class='finding-badge'>⚠️ {f}</div>",
                unsafe_allow_html=True,
            )
    else:
        st.success("🎉 No major issues found! Your CV is well-optimised.")

    # Re-run button
    st.divider()
    if st.button("🔄 Re-run Analysis", use_container_width=True):
        _run_analysis()
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — JD Match
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("## 🎯 Job Description Match")

    jd_match = st.session_state.get("jd_match", {})

    if not st.session_state["jd_text"]:
        st.info("Paste a job description in the sidebar to see your match score and skills gap.")
    elif not jd_match:
        st.warning("Something went wrong with the analysis. Try re-uploading your CV.")
    else:
        sim   = jd_match.get("similarity_score", 0)
        cov   = jd_match.get("keyword_coverage", 0)
        color = _score_color(int(sim))

        # Match score
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(
                f"<h1 style='font-size:3rem; color:{color}'>{sim}%</h1>"
                f"<p style='color:#6b7280'>Overall Match</p>",
                unsafe_allow_html=True,
            )
            st.metric("Keyword Coverage", f"{cov}%")

        with col2:
            st.markdown("### Match interpretation")
            if sim >= 75:
                st.success("**Strong match** — your CV aligns well with this role. Apply with confidence.")
            elif sim >= 55:
                st.warning("**Moderate match** — address the skills gap below before applying.")
            elif sim >= 35:
                st.warning("**Weak match** — significant work needed; focus on the missing keywords.")
            else:
                st.error("**Very low match** — this role may be a stretch. Consider upskilling first.")

        st.divider()

        # Skills analysis
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            matched = jd_match.get("matched_skills", [])
            st.markdown(f"### ✅ Matched Skills ({len(matched)})")
            if matched:
                for s in matched:
                    st.markdown(f"- {s}")
            else:
                st.caption("No common skills detected")

        with col_b:
            missing = jd_match.get("missing_skills", [])
            st.markdown(f"### ❌ Missing Skills ({len(missing)})")
            if missing:
                for s in missing:
                    st.markdown(f"- **{s}**")
            else:
                st.caption("No obvious gaps detected")

        with col_c:
            extra = jd_match.get("extra_skills", [])
            st.markdown(f"### 💡 Bonus Skills ({len(extra)})")
            if extra:
                for s in extra[:10]:
                    st.markdown(f"- {s}")
            else:
                st.caption("—")

        st.divider()

        # Top JD keywords
        top_kws = jd_match.get("top_jd_keywords", [])
        if top_kws:
            st.markdown("### 🏷️ Top JD Keywords")
            st.caption("Make sure these words appear in your CV")
            kw_html = " ".join(
                f"<span style='background:#dbeafe; color:#1e40af; padding:3px 9px; "
                f"border-radius:99px; margin:3px; font-size:0.85rem; display:inline-block'>"
                f"{kw}</span>"
                for kw in top_kws[:20]
            )
            st.markdown(kw_html, unsafe_allow_html=True)

        # AI gap analysis
        st.divider()
        if is_configured():
            if st.button("🤖 Get AI Skills Gap Analysis", use_container_width=True):
                with st.spinner("Analysing skills gap with Gemini…"):
                    try:
                        gap = analyse_skills_gap(
                            st.session_state["cv_text"],
                            st.session_state["jd_text"],
                            missing,
                        )
                        st.markdown("### 🧠 AI Gap Analysis")
                        st.write(gap.get("gap_analysis", ""))

                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.markdown("**⚡ Quick Wins**")
                            for item in gap.get("quick_wins", []):
                                st.markdown(f"- {item}")
                        with c2:
                            st.markdown("**📚 Learning Paths**")
                            for item in gap.get("learning_paths", []):
                                st.markdown(f"- {item}")
                        with c3:
                            st.markdown("**🔄 Transferable**")
                            for item in gap.get("transferable", []):
                                st.markdown(f"- {item}")
                    except Exception as e:
                        st.error(f"AI analysis failed: {e}")
        else:
            st.info("Add your Gemini API key to enable AI-powered gap analysis.")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Bullet Rewriter
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("## ✏️ AI Bullet Rewriter")

    if not is_configured():
        st.warning("Add your Gemini API key to Streamlit secrets to use the AI rewriter.", icon="🔑")
        st.stop()

    # Options row
    col_opt1, col_opt2, col_opt3 = st.columns(3)
    with col_opt1:
        tone = st.selectbox(
            "Rewriting tone",
            ["professional", "concise", "achievement-focused"],
            index=0,
        )
    with col_opt2:
        use_jd = st.checkbox(
            "Optimise for job description",
            value=bool(st.session_state["jd_text"]),
            disabled=not bool(st.session_state["jd_text"]),
        )
    with col_opt3:
        st.markdown("&nbsp;", unsafe_allow_html=True)  # spacer

    st.divider()

    # Extract bullets from experience section
    exp_text = st.session_state["cv_sections"].get("experience", "")
    bullets  = _extract_bullets(exp_text)

    if not bullets:
        # Fall back to all bullets in raw text
        bullets = _extract_bullets(st.session_state["cv_text"])

    if not bullets:
        st.info("No bullet points found in your CV. Make sure your experience section uses bullet characters (•, -, *, ▸).")
    else:
        st.markdown(f"**{len(bullets)} bullet point(s) detected**")

        # Batch rewrite all button
        if st.button(f"⚡ Rewrite All {len(bullets)} Bullets", use_container_width=True, type="primary"):
            with st.spinner(f"Rewriting {len(bullets)} bullets with Gemini…"):
                try:
                    jd_ctx = st.session_state["jd_text"] if use_jd else ""
                    results = rewrite_bullets_batch(bullets[:20], jd_ctx)
                    for r in results:
                        st.session_state["rewrites"][r["original"]] = r["rewritten"]
                    st.success(f"✅ Rewrote {len(results)} bullets!")
                except Exception as e:
                    st.error(f"Rewriting failed: {e}")

        st.divider()

        # Individual bullet cards
        jd_ctx = st.session_state["jd_text"] if use_jd else ""

        for i, bullet in enumerate(bullets[:20]):
            rewritten = st.session_state["rewrites"].get(bullet, "")
            accepted  = bullet in st.session_state["accepted"]

            with st.expander(
                f"{'✅' if accepted else '📝'} Bullet {i+1}: {bullet[:70]}{'…' if len(bullet) > 70 else ''}",
                expanded=(not rewritten),
            ):
                st.markdown("**Original:**")
                st.markdown(f"> {bullet}")

                if rewritten:
                    st.markdown("**Rewritten:**")
                    st.markdown(
                        f"<div style='background:#f0fdf4; border-left:3px solid #22c55e; "
                        f"padding:10px 14px; border-radius:0 8px 8px 0; margin:8px 0'>"
                        f"✨ {rewritten}</div>",
                        unsafe_allow_html=True,
                    )

                    col_a, col_b, col_c = st.columns([1, 1, 2])
                    with col_a:
                        if st.button("✅ Accept", key=f"accept_{i}"):
                            st.session_state["accepted"].add(bullet)
                            st.rerun()
                    with col_b:
                        if st.button("🔄 Retry", key=f"retry_{i}"):
                            with st.spinner("Rewriting…"):
                                try:
                                    new = rewrite_bullet(bullet, jd_ctx, tone)
                                    st.session_state["rewrites"][bullet] = new
                                    st.rerun()
                                except Exception as e:
                                    st.error(str(e))
                    with col_c:
                        if accepted:
                            if st.button("↩️ Undo Accept", key=f"undo_{i}"):
                                st.session_state["accepted"].discard(bullet)
                                st.rerun()
                else:
                    if st.button("✨ Rewrite this bullet", key=f"rewrite_{i}"):
                        with st.spinner("Rewriting…"):
                            try:
                                new = rewrite_bullet(bullet, jd_ctx, tone)
                                st.session_state["rewrites"][bullet] = new
                                st.rerun()
                            except Exception as e:
                                st.error(str(e))

        # Accepted summary
        if st.session_state["accepted"]:
            st.divider()
            st.markdown(f"### ✅ Accepted Rewrites ({len(st.session_state['accepted'])})")
            for orig in st.session_state["accepted"]:
                new = st.session_state["rewrites"].get(orig, orig)
                st.markdown(f"~~{orig}~~")
                st.markdown(f"→ **{new}**")
                st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — AI Tools
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("## 💬 AI Tools")

    if not is_configured():
        st.warning("Add your Gemini API key to Streamlit secrets to use AI tools.", icon="🔑")
        st.stop()

    ai_tool = st.selectbox(
        "Choose an AI tool",
        [
            "📝 CV Review",
            "📋 Professional Summary Generator",
            "✉️ Cover Letter Generator",
            "🎤 Interview Questions Predictor",
        ],
    )

    st.divider()

    # ── CV Review ──────────────────────────────────────────────────────
    if ai_tool == "📝 CV Review":
        st.markdown("### 📝 Full CV Review")
        st.caption("Gemini will review your CV and provide structured feedback.")

        ats_total = st.session_state.get("ats_result", {}).get("total", 0)
        jd_sim    = st.session_state.get("jd_match", {}).get("similarity_score", 0)

        if st.button("🤖 Generate CV Review", use_container_width=True, type="primary"):
            with st.spinner("Generating review with Gemini…"):
                try:
                    review = full_cv_review(
                        st.session_state["cv_text"],
                        ats_total,
                        jd_sim,
                    )
                    st.markdown(review)
                except Exception as e:
                    st.error(f"Review failed: {e}")

    # ── Summary Generator ───────────────────────────────────────────────
    elif ai_tool == "📋 Professional Summary Generator":
        st.markdown("### 📋 Professional Summary Generator")
        st.caption("Generate a keyword-rich professional summary tailored to your target role.")

        col1, col2 = st.columns(2)
        with col1:
            word_target = st.slider("Target word count", 60, 120, 80, step=10)
        with col2:
            use_jd_sum = st.checkbox("Tailor to job description", value=bool(st.session_state["jd_text"]))

        if st.button("✨ Generate Summary", use_container_width=True, type="primary"):
            with st.spinner("Generating summary…"):
                try:
                    jd_ctx = st.session_state["jd_text"] if use_jd_sum else ""
                    summary = generate_summary(st.session_state["cv_text"], jd_ctx, word_target)
                    st.markdown("### Generated Summary")
                    st.markdown(
                        f"<div style='background:#f8fafc; border:1px solid #e2e8f0; "
                        f"border-radius:10px; padding:16px; font-size:1rem; line-height:1.7'>"
                        f"{summary}</div>",
                        unsafe_allow_html=True,
                    )
                    st.markdown(f"*{len(summary.split())} words*")
                    st.download_button(
                        "📋 Copy / Download Summary",
                        data=summary,
                        file_name="professional_summary.txt",
                        mime="text/plain",
                    )
                except Exception as e:
                    st.error(f"Summary generation failed: {e}")

    # ── Cover Letter ────────────────────────────────────────────────────
    elif ai_tool == "✉️ Cover Letter Generator":
        st.markdown("### ✉️ Cover Letter Generator")
        if not st.session_state["jd_text"]:
            st.warning("Paste a job description in the sidebar for a tailored cover letter.")

        col1, col2 = st.columns(2)
        with col1:
            company_name = st.text_input("Company name", placeholder="e.g. Acme Corp")
        with col2:
            role_title = st.text_input("Role title", placeholder="e.g. Senior Data Analyst")

        cl_tone = st.selectbox("Tone", ["professional", "enthusiastic", "formal", "conversational"])

        if st.button("✉️ Generate Cover Letter", use_container_width=True, type="primary"):
            with st.spinner("Writing cover letter…"):
                try:
                    letter = generate_cover_letter(
                        st.session_state["cv_text"],
                        st.session_state["jd_text"],
                        company_name,
                        role_title,
                        cl_tone,
                    )
                    st.markdown("### Cover Letter")
                    st.text_area("", value=letter, height=350)
                    st.download_button(
                        "📥 Download Cover Letter",
                        data=letter,
                        file_name=f"cover_letter_{company_name or 'company'}.txt",
                        mime="text/plain",
                    )
                except Exception as e:
                    st.error(f"Cover letter generation failed: {e}")

    # ── Interview Questions ─────────────────────────────────────────────
    elif ai_tool == "🎤 Interview Questions Predictor":
        st.markdown("### 🎤 Interview Questions Predictor")
        if not st.session_state["jd_text"]:
            st.info("Add a job description for more targeted questions.")

        n_questions = st.slider("Number of questions", 5, 12, 8)

        if st.button("🎤 Predict Interview Questions", use_container_width=True, type="primary"):
            with st.spinner("Predicting questions with Gemini…"):
                try:
                    questions = predict_interview_questions(
                        st.session_state["cv_text"],
                        st.session_state["jd_text"],
                        n_questions,
                    )
                    type_colors = {
                        "Behavioural":  "#dbeafe",
                        "Technical":    "#d1fae5",
                        "Situational":  "#fef3c7",
                        "Motivational": "#ede9fe",
                    }
                    for i, q in enumerate(questions, 1):
                        q_type  = q.get("type", "General")
                        bg      = type_colors.get(q_type, "#f3f4f6")
                        tip     = q.get("tip", "")
                        question_text = q.get("question", "")

                        st.markdown(
                            f"<div style='background:{bg}; border-radius:10px; "
                            f"padding:14px 18px; margin:10px 0'>"
                            f"<strong>Q{i} [{q_type}]</strong><br>"
                            f"{question_text}"
                            f"{'<br><span style=\"color:#6b7280; font-size:0.88rem\">💡 ' + tip + '</span>' if tip else ''}"
                            f"</div>",
                            unsafe_allow_html=True,
                        )
                except Exception as e:
                    st.error(f"Prediction failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — Export
# ─────────────────────────────────────────────────────────────────────────────
with tab5:
    st.markdown("## 📥 Export")

    accepted_rewrites = {
        orig: st.session_state["rewrites"][orig]
        for orig in st.session_state["accepted"]
        if orig in st.session_state["rewrites"]
    }

    n_accepted = len(accepted_rewrites)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accepted bullet rewrites", n_accepted)
    with col2:
        ats_total = st.session_state.get("ats_result", {}).get("total", 0)
        st.metric("ATS Score", f"{ats_total}/100")

    st.divider()

    # ── Download optimised CV as PDF ────────────────────────────────────
    st.markdown("### 📄 Download Optimised CV (PDF)")
    if n_accepted == 0:
        st.info("Accept some bullet rewrites in the **Rewriter** tab to apply them to your exported PDF.")
    else:
        st.success(f"✅ {n_accepted} rewritten bullet(s) will be applied to the PDF.")

    if st.button("🔨 Generate PDF", use_container_width=True, type="primary"):
        with st.spinner("Building PDF…"):
            try:
                pdf_bytes = generate_pdf(
                    st.session_state["cv_sections"],
                    st.session_state["contact_info"],
                    rewrites=accepted_rewrites if accepted_rewrites else None,
                )
                fname = st.session_state["file_name"].replace(".pdf", "") or "cv"
                st.download_button(
                    label="📥 Download Optimised CV PDF",
                    data=pdf_bytes,
                    file_name=f"{fname}_optimized.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
                st.success("PDF ready! Click the button above to download.")
            except ImportError as e:
                st.error(f"PDF export requires reportlab: {e}")
            except Exception as e:
                st.error(f"PDF generation failed: {e}")

    st.divider()

    # ── Download as plain text ──────────────────────────────────────────
    st.markdown("### 📋 Download as Plain Text")
    st.caption("Useful for pasting into ATS portals that don't accept PDF formatting.")

    cv_text_export = st.session_state["cv_text"]
    if accepted_rewrites:
        for orig, new in accepted_rewrites.items():
            cv_text_export = cv_text_export.replace(orig, new)

    st.download_button(
        label="📥 Download CV as .txt",
        data=cv_text_export,
        file_name="cv_optimized.txt",
        mime="text/plain",
        use_container_width=True,
    )

    st.divider()

    # ── ATS Report ──────────────────────────────────────────────────────
    st.markdown("### 📊 Download ATS Report")
    ats_result = st.session_state.get("ats_result", {})
    jd_match   = st.session_state.get("jd_match",   {})

    if ats_result:
        report_lines = [
            "CV OPTIMIZER — ATS REPORT",
            "=" * 40,
            f"ATS Score: {ats_result.get('total', 0)}/100 ({ats_result.get('grade', '')})",
            "",
        ]
        if jd_match:
            report_lines += [
                f"JD Match Score: {jd_match.get('similarity_score', 0)}%",
                f"Keyword Coverage: {jd_match.get('keyword_coverage', 0)}%",
                "",
                "Missing Skills:",
                *[f"  - {s}" for s in jd_match.get("missing_skills", [])],
                "",
            ]
        report_lines += [
            "Dimension Scores:",
            *[
                f"  {name}: {data['score']}/{data['max']}"
                for name, data in ats_result.get("dimensions", {}).items()
            ],
            "",
            "Recommendations:",
            *[f"  ⚠ {f}" for f in ats_result.get("all_findings", [])],
        ]
        if accepted_rewrites:
            report_lines += [
                "",
                f"Accepted Bullet Rewrites ({n_accepted}):",
                *[
                    f"  ORIGINAL: {orig}\n  REWRITTEN: {new}\n"
                    for orig, new in accepted_rewrites.items()
                ],
            ]

        report_text = "\n".join(report_lines)
        st.download_button(
            label="📥 Download ATS Report (.txt)",
            data=report_text,
            file_name="ats_report.txt",
            mime="text/plain",
            use_container_width=True,
        )
    else:
        st.info("Upload a CV to generate an ATS report.")
