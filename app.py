"""
app.py — AI-Powered CV Optimizer
Built with Streamlit + Gemini 2.5 Flash
"""
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import time

from utils.cv_parser import parse_cv, extract_contact_info
from utils.ats_scorer import compute_ats_score, extract_jd_keywords
from utils.jd_matcher import match_cv_to_jd
from utils.gemini_client import (
    analyse_cv_overall,
    rewrite_bullet,
    generate_summary,
    suggest_missing_skills,
    tailor_cv_for_jd,
)
from utils.pdf_exporter import export_cv_to_pdf


# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CV Optimizer AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main accent */
    :root { --accent: #6C63FF; }

    .stApp { background-color: #0F1117; }

    /* Score gauge card */
    .score-card {
        background: linear-gradient(135deg, #1C1E2E, #16182A);
        border: 1px solid #2E3058;
        border-radius: 16px;
        padding: 24px;
        text-align: center;
        margin-bottom: 16px;
    }
    .score-number {
        font-size: 64px;
        font-weight: 800;
        background: linear-gradient(135deg, #6C63FF, #A78BFA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1;
    }
    .score-label {
        color: #8B8FA8;
        font-size: 13px;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }

    /* Diff view */
    .original-text {
        background: rgba(239,68,68,0.08);
        border-left: 3px solid #EF4444;
        border-radius: 0 8px 8px 0;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 14px;
        color: #FCA5A5;
        line-height: 1.5;
    }
    .rewritten-text {
        background: rgba(34,197,94,0.08);
        border-left: 3px solid #22C55E;
        border-radius: 0 8px 8px 0;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 14px;
        color: #86EFAC;
        line-height: 1.5;
    }

    /* Section badge */
    .section-badge {
        display: inline-block;
        background: rgba(108,99,255,0.15);
        color: #A78BFA;
        border: 1px solid rgba(108,99,255,0.3);
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 12px;
    }

    /* Skill pill */
    .skill-match { background:rgba(34,197,94,0.12); color:#86EFAC; border:1px solid rgba(34,197,94,0.25); border-radius:12px; padding:3px 10px; font-size:12px; display:inline-block; margin:2px; }
    .skill-miss  { background:rgba(239,68,68,0.12);  color:#FCA5A5; border:1px solid rgba(239,68,68,0.25);  border-radius:12px; padding:3px 10px; font-size:12px; display:inline-block; margin:2px; }

    /* Warning / suggestion boxes */
    .warn-box { background:rgba(251,191,36,0.08); border:1px solid rgba(251,191,36,0.25); border-radius:8px; padding:10px 14px; margin:4px 0; font-size:13px; color:#FCD34D; }
    .sugg-box { background:rgba(108,99,255,0.08); border:1px solid rgba(108,99,255,0.25); border-radius:8px; padding:10px 14px; margin:4px 0; font-size:13px; color:#C4B5FD; }

    /* Hide Streamlit chrome */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 1rem; }

    /* Tab styling */
    .stTabs [data-baseweb="tab"] { font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ─────────────────────────────────────────────
def init_state():
    defaults = {
        "parsed_cv": None,
        "contact_info": {},
        "ats_score": None,
        "jd_match": None,
        "rewrites": {},          # section -> list of {original, rewritten, accepted}
        "cv_analysis": "",
        "summary_draft": "",
        "skill_suggestions": "",
        "jd_text": "",
        "active_tab": 0,
        "version_history": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── Helpers ──────────────────────────────────────────────────────────────────
def score_color(score: int) -> str:
    if score >= 75: return "#22C55E"
    if score >= 50: return "#F59E0B"
    return "#EF4444"


def match_color(score: float) -> str:
    if score >= 0.65: return "#22C55E"
    if score >= 0.40: return "#F59E0B"
    return "#EF4444"


def render_ats_gauge(score: int):
    color = score_color(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#8B8FA8", "tickfont": {"color": "#8B8FA8"}},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "#1C1E2E",
            "bordercolor": "#2E3058",
            "steps": [
                {"range": [0, 40],  "color": "rgba(239,68,68,0.12)"},
                {"range": [40, 70], "color": "rgba(245,158,11,0.12)"},
                {"range": [70, 100],"color": "rgba(34,197,94,0.12)"},
            ],
            "threshold": {"line": {"color": color, "width": 3}, "value": score},
        },
        number={"font": {"size": 48, "color": color}, "suffix": "/100"},
    ))
    fig.update_layout(
        height=220,
        margin=dict(l=20, r=20, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font_color="#FAFAFA",
    )
    st.plotly_chart(fig, use_container_width=True)


def render_breakdown_chart(breakdown: dict):
    labels = list(breakdown.keys())
    scored = [v[0] for v in breakdown.values()]
    maxes  = [v[1] for v in breakdown.values()]
    pct    = [s/m*100 for s, m in zip(scored, maxes)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=maxes, orientation="h",
        marker_color="rgba(46,48,88,0.8)", name="Max",
        showlegend=False,
    ))
    fig.add_trace(go.Bar(
        y=labels,
        x=scored,
        orientation="h",
        marker_color=["#22C55E" if p >= 70 else "#F59E0B" if p >= 40 else "#EF4444" for p in pct],
        name="Score",
        text=[f"{s}/{m}" for s, m in zip(scored, maxes)],
        textposition="inside",
        insidetextanchor="start",
        showlegend=False,
    ))
    fig.update_layout(
        barmode="overlay",
        height=220,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#FAFAFA",
        xaxis=dict(showgrid=False, showticklabels=False),
        yaxis=dict(tickfont=dict(size=11)),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_jd_heatmap(heatmap: dict, cv_text: str):
    if not heatmap:
        return
    terms = list(heatmap.keys())[:20]
    freqs = [heatmap[t] for t in terms]
    cv_lower = cv_text.lower()
    in_cv = ["#22C55E" if t in cv_lower else "#EF4444" for t in terms]

    fig = go.Figure(go.Bar(
        x=freqs, y=terms, orientation="h",
        marker_color=in_cv,
        text=["✓ in CV" if t in cv_lower else "✗ missing" for t in terms],
        textposition="outside",
    ))
    fig.update_layout(
        height=420,
        margin=dict(l=10, r=90, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#FAFAFA",
        xaxis=dict(showgrid=False, title="Frequency in JD"),
        yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🚀 CV Optimizer AI")
    st.markdown("<p style='color:#8B8FA8;font-size:13px;'>Powered by Gemini 2.5 Flash</p>", unsafe_allow_html=True)
    st.divider()

    # ── CV Upload
    st.markdown("### 📄 Upload CV")
    uploaded = st.file_uploader("Upload your CV (PDF)", type=["pdf"], label_visibility="collapsed")

    if uploaded and (
        st.session_state.parsed_cv is None
        or st.session_state.get("last_upload") != uploaded.name
    ):
        with st.spinner("Parsing CV..."):
            file_bytes = uploaded.read()
            st.session_state.parsed_cv = parse_cv(file_bytes)
            st.session_state.contact_info = extract_contact_info(st.session_state.parsed_cv)
            st.session_state.last_upload = uploaded.name
            st.session_state.rewrites = {}
            st.session_state.cv_analysis = ""
            st.session_state.summary_draft = ""
            st.session_state.ats_score = None
            st.session_state.jd_match = None
        st.success(f"✓ Parsed — {st.session_state.parsed_cv.page_count} page(s), "
                   f"{len(st.session_state.parsed_cv.sections)} sections found")

    st.divider()

    # ── Job Description
    st.markdown("### 🎯 Job Description")
    jd_input = st.text_area(
        "Paste the job description",
        value=st.session_state.jd_text,
        height=180,
        placeholder="Paste the full job description here for keyword matching and tailoring...",
        label_visibility="collapsed",
    )
    if jd_input != st.session_state.jd_text:
        st.session_state.jd_text = jd_input
        st.session_state.jd_match = None
        st.session_state.ats_score = None

    st.divider()

    # ── Analyse button
    analyse_disabled = st.session_state.parsed_cv is None
    if st.button("⚡ Analyse CV", use_container_width=True, disabled=analyse_disabled, type="primary"):
        cv = st.session_state.parsed_cv
        jd = st.session_state.jd_text

        with st.spinner("Running ATS analysis..."):
            st.session_state.ats_score = compute_ats_score(
                cv.full_text, cv.sections, jd
            )

        with st.spinner("Matching against job description..."):
            st.session_state.jd_match = match_cv_to_jd(cv.sections, jd)

        with st.spinner("Generating AI analysis..."):
            ats_total = st.session_state.ats_score.total
            st.session_state.cv_analysis = analyse_cv_overall(cv.full_text, jd, ats_total)

        st.success("Analysis complete!")

    if st.session_state.parsed_cv:
        st.divider()
        st.markdown("### ℹ️ CV Info")
        cv = st.session_state.parsed_cv
        st.markdown(f"**Pages:** {cv.page_count}")
        st.markdown(f"**Sections:** {len(cv.sections)}")
        st.markdown(f"**Words:** {len(cv.full_text.split())}")
        detected = ", ".join(cv.sections.keys())
        st.markdown(f"**Detected:** `{detected}`")


# ── Main content ─────────────────────────────────────────────────────────────
if st.session_state.parsed_cv is None:
    # ── Welcome screen
    st.markdown("""
    <div style='text-align:center; padding: 60px 20px;'>
        <div style='font-size:72px; margin-bottom:16px;'>🚀</div>
        <h1 style='font-size:36px; font-weight:800; margin-bottom:8px;'>CV Optimizer AI</h1>
        <p style='color:#8B8FA8; font-size:16px; max-width:520px; margin:0 auto 32px;'>
            Upload your CV and a job description to get AI-powered rewrites, ATS scoring,
            keyword gap analysis, and a side-by-side live diff editor.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    features = [
        ("🎯", "ATS Scoring", "100-point compliance score with section-by-section breakdown"),
        ("🔍", "JD Matching", "Semantic keyword analysis and skills gap heatmap"),
        ("✍️", "AI Rewrites", "Gemini 2.5 Flash rewrites bullets in STAR format"),
        ("📄", "PDF Export", "Download your optimised CV as a clean, ATS-ready PDF"),
    ]
    for col, (icon, title, desc) in zip([col1, col2, col3, col4], features):
        with col:
            st.markdown(f"""
            <div style='background:#1C1E2E; border:1px solid #2E3058; border-radius:12px; padding:20px; text-align:center; height:160px;'>
                <div style='font-size:32px; margin-bottom:8px;'>{icon}</div>
                <div style='font-weight:700; margin-bottom:6px;'>{title}</div>
                <div style='color:#8B8FA8; font-size:12px;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👈 Upload your CV PDF in the sidebar to get started")
    st.stop()


cv = st.session_state.parsed_cv
jd = st.session_state.jd_text
contact = st.session_state.contact_info

# ── Name header
name = contact.get("author") or cv.metadata.get("author") or "Your Name"
st.markdown(f"<h2 style='margin-bottom:4px;'>{name}</h2>", unsafe_allow_html=True)
if contact.get("email"):
    st.markdown(f"<p style='color:#8B8FA8; margin-top:0;'>{contact.get('email', '')}  |  {contact.get('phone', '')}  |  {contact.get('linkedin', '')}</p>", unsafe_allow_html=True)

st.divider()

# ── Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard",
    "✍️ Live Editor",
    "🎯 JD Match",
    "🤖 AI Tools",
    "📄 Export",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    if st.session_state.ats_score is None:
        st.info("Click **⚡ Analyse CV** in the sidebar to run the full analysis.")
    else:
        ats = st.session_state.ats_score
        jdm = st.session_state.jd_match

        # ── Row 1: Score cards
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            c = score_color(ats.total)
            st.markdown(f"""
            <div class='score-card'>
                <div class='score-number' style='background:linear-gradient(135deg,{c},{c}88);-webkit-background-clip:text;'>{ats.total}</div>
                <div class='score-label'>ATS Score</div>
            </div>""", unsafe_allow_html=True)

        with col2:
            wc = ats.word_count
            c2 = "#22C55E" if 400 <= wc <= 900 else "#F59E0B"
            st.markdown(f"""
            <div class='score-card'>
                <div class='score-number' style='background:linear-gradient(135deg,{c2},{c2}88);-webkit-background-clip:text;'>{wc}</div>
                <div class='score-label'>Word Count</div>
            </div>""", unsafe_allow_html=True)

        with col3:
            pv = len(ats.found_power_verbs)
            c3 = "#22C55E" if pv >= 8 else "#F59E0B" if pv >= 4 else "#EF4444"
            st.markdown(f"""
            <div class='score-card'>
                <div class='score-number' style='background:linear-gradient(135deg,{c3},{c3}88);-webkit-background-clip:text;'>{pv}</div>
                <div class='score-label'>Power Verbs</div>
            </div>""", unsafe_allow_html=True)

        with col4:
            jd_pct = int((jdm.overall_score if jdm else 0) * 100)
            c4 = match_color(jdm.overall_score if jdm else 0)
            st.markdown(f"""
            <div class='score-card'>
                <div class='score-number' style='background:linear-gradient(135deg,{c4},{c4}88);-webkit-background-clip:text;'>{jd_pct}%</div>
                <div class='score-label'>JD Match</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Row 2: Gauge + Breakdown
        col_g, col_b = st.columns([1, 1])
        with col_g:
            st.markdown("#### ATS Score Gauge")
            render_ats_gauge(ats.total)
        with col_b:
            st.markdown("#### Score Breakdown")
            render_breakdown_chart(ats.breakdown)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Row 3: Warnings + Suggestions
        col_w, col_s = st.columns(2)
        with col_w:
            st.markdown("#### ⚠️ Warnings")
            if ats.warnings:
                for w in ats.warnings:
                    st.markdown(f"<div class='warn-box'>⚠️ {w}</div>", unsafe_allow_html=True)
            else:
                st.success("No critical warnings — great structure!")

        with col_s:
            st.markdown("#### 💡 Suggestions")
            for s in ats.suggestions[:6]:
                st.markdown(f"<div class='sugg-box'>💡 {s}</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Row 4: Power verbs found
        if ats.found_power_verbs:
            st.markdown("#### ✅ Power Verbs Detected")
            pills = "".join(f"<span class='skill-match'>{v}</span>" for v in ats.found_power_verbs)
            st.markdown(pills, unsafe_allow_html=True)

        if ats.weak_verb_hits:
            st.markdown("#### ❌ Weak Verbs to Replace")
            pills = "".join(f"<span class='skill-miss'>{v}</span>" for v in ats.weak_verb_hits)
            st.markdown(pills, unsafe_allow_html=True)

        # ── AI Analysis
        if st.session_state.cv_analysis:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("#### 🤖 AI Coach Analysis")
            with st.expander("View full analysis", expanded=True):
                st.markdown(st.session_state.cv_analysis)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: LIVE EDITOR (Side-by-side diff)
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### ✍️ Live CV Editor")
    st.markdown("<p style='color:#8B8FA8;'>AI-powered bullet rewriter. Accept or reject each suggestion.</p>", unsafe_allow_html=True)

    sections_with_bullets = {
        k: v for k, v in cv.sections.items()
        if v.bullets and k not in ("contact",)
    }

    if not sections_with_bullets:
        st.warning("No sections with bullet points found. Your CV may not have a standard bullet format.")
    else:
        selected_section = st.selectbox(
            "Select section to rewrite",
            options=list(sections_with_bullets.keys()),
            format_func=lambda x: x.title(),
        )

        section = sections_with_bullets[selected_section]
        bullets = section.bullets

        col_rewrite, col_options = st.columns([3, 1])
        with col_options:
            tone = st.selectbox("Tone", ["professional", "technical", "concise", "impactful"])

        with col_rewrite:
            if st.button(f"🤖 Rewrite all {selected_section.title()} bullets", type="primary"):
                st.session_state.rewrites[selected_section] = []
                progress = st.progress(0, text="Rewriting bullets...")
                for i, bullet in enumerate(bullets):
                    if len(bullet.strip()) < 10:
                        continue
                    with st.spinner(f"Rewriting bullet {i+1}/{len(bullets)}..."):
                        rewritten = rewrite_bullet(bullet, selected_section, jd, tone)
                    st.session_state.rewrites[selected_section].append({
                        "original": bullet,
                        "rewritten": rewritten,
                        "accepted": False,
                        "skipped": False,
                    })
                    progress.progress((i + 1) / len(bullets), text=f"Rewriting {i+1}/{len(bullets)}...")
                progress.empty()
                st.rerun()

        # ── Show diff
        if selected_section in st.session_state.rewrites:
            rewrites = st.session_state.rewrites[selected_section]
            accepted_count = sum(1 for r in rewrites if r["accepted"])
            st.markdown(f"**{accepted_count}/{len(rewrites)} accepted**")

            # Bulk actions
            bc1, bc2, _ = st.columns([1, 1, 3])
            with bc1:
                if st.button("✅ Accept All"):
                    for r in rewrites:
                        r["accepted"] = True
                    st.rerun()
            with bc2:
                if st.button("❌ Reject All"):
                    for r in rewrites:
                        r["accepted"] = False
                    st.rerun()

            st.markdown("---")

            for i, item in enumerate(rewrites):
                st.markdown(f"<div class='section-badge'>Bullet {i+1}</div>", unsafe_allow_html=True)

                st.markdown("<div class='original-text'>🔴 <strong>Original</strong><br>" + item["original"] + "</div>", unsafe_allow_html=True)
                st.markdown("<div class='rewritten-text'>🟢 <strong>AI Rewrite</strong><br>" + item["rewritten"] + "</div>", unsafe_allow_html=True)

                col_a, col_r, col_e, _ = st.columns([1, 1, 1, 3])
                with col_a:
                    if st.button("✅ Accept", key=f"accept_{selected_section}_{i}"):
                        rewrites[i]["accepted"] = True
                        st.rerun()
                with col_r:
                    if st.button("❌ Reject", key=f"reject_{selected_section}_{i}"):
                        rewrites[i]["accepted"] = False
                        st.rerun()
                with col_e:
                    if st.button("✏️ Edit", key=f"edit_{selected_section}_{i}"):
                        st.session_state[f"editing_{selected_section}_{i}"] = True

                # Inline editing
                if st.session_state.get(f"editing_{selected_section}_{i}"):
                    edited = st.text_area(
                        "Edit rewrite",
                        value=rewrites[i]["rewritten"],
                        key=f"edit_area_{selected_section}_{i}",
                        height=80,
                    )
                    if st.button("💾 Save Edit", key=f"save_{selected_section}_{i}"):
                        rewrites[i]["rewritten"] = edited
                        rewrites[i]["accepted"] = True
                        st.session_state[f"editing_{selected_section}_{i}"] = False
                        st.rerun()

                if item["accepted"]:
                    st.success("✅ Accepted")
                st.markdown("<br>", unsafe_allow_html=True)

        else:
            # Show original bullets
            st.markdown("**Original bullets:**")
            for b in bullets:
                st.markdown(f"• {b}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: JD MATCH
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### 🎯 Job Description Match Analysis")

    if not jd:
        st.info("Paste a job description in the sidebar and click **⚡ Analyse CV** to see matching.")
    elif st.session_state.jd_match is None:
        st.info("Click **⚡ Analyse CV** in the sidebar to run JD matching.")
    else:
        jdm = st.session_state.jd_match

        # Overall match score
        pct = int(jdm.overall_score * 100)
        col_score, col_info = st.columns([1, 2])
        with col_score:
            c = match_color(jdm.overall_score)
            st.markdown(f"""
            <div class='score-card' style='padding:32px;'>
                <div style='font-size:72px; font-weight:800; color:{c};'>{pct}%</div>
                <div class='score-label'>Overall JD Match</div>
            </div>""", unsafe_allow_html=True)

        with col_info:
            st.markdown("#### Matched Skills")
            if jdm.top_matched_skills:
                pills = "".join(f"<span class='skill-match'>✓ {s}</span>" for s in jdm.top_matched_skills)
                st.markdown(pills, unsafe_allow_html=True)
            else:
                st.markdown("<span style='color:#8B8FA8'>No tech skills matched yet</span>", unsafe_allow_html=True)

            st.markdown("#### Missing Skills")
            if jdm.top_missing_skills:
                pills = "".join(f"<span class='skill-miss'>✗ {s}</span>" for s in jdm.top_missing_skills)
                st.markdown(pills, unsafe_allow_html=True)
            else:
                st.success("All detected skills are present in your CV!")

        st.markdown("---")

        # Required vs Nice-to-have
        if jdm.required_skills or jdm.nice_to_have_skills:
            col_r, col_n = st.columns(2)
            with col_r:
                st.markdown("#### 🔴 Required Skills")
                cv_lower = cv.full_text.lower()
                for s in jdm.required_skills:
                    present = s in cv_lower
                    icon = "✅" if present else "❌"
                    st.markdown(f"{icon} `{s}`")
            with col_n:
                st.markdown("#### 🟡 Nice to Have")
                for s in jdm.nice_to_have_skills:
                    present = s in cv_lower
                    icon = "✅" if present else "○"
                    st.markdown(f"{icon} `{s}`")

        st.markdown("---")

        # Section-by-section scores
        st.markdown("#### Section Match Scores")
        if jdm.section_scores:
            df = pd.DataFrame([
                {
                    "Section": s.section.title(),
                    "Match Score": f"{int(s.score * 100)}%",
                    "Score (raw)": s.score,
                    "Key Matches": ", ".join(s.matched_terms[:5]),
                }
                for s in jdm.section_scores
            ])
            st.dataframe(
                df[["Section", "Match Score", "Key Matches"]],
                use_container_width=True,
                hide_index=True,
            )

        st.markdown("---")

        # Keyword heatmap
        st.markdown("#### 🌡️ JD Keyword Heatmap")
        st.markdown("<p style='color:#8B8FA8; font-size:13px;'>Green = found in your CV. Red = missing.</p>", unsafe_allow_html=True)
        render_jd_heatmap(jdm.keyword_heatmap, cv.full_text)

        # Skill gap suggestions
        st.markdown("---")
        st.markdown("#### 💡 How to Address Skill Gaps")
        if st.session_state.skill_suggestions:
            st.markdown(st.session_state.skill_suggestions)
        elif jdm.top_missing_skills:
            if st.button("🤖 Get AI suggestions for skill gaps"):
                with st.spinner("Generating suggestions..."):
                    st.session_state.skill_suggestions = suggest_missing_skills(
                        cv.full_text, jd, jdm.top_missing_skills
                    )
                st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: AI TOOLS
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("### 🤖 AI-Powered Tools")

    tool = st.radio(
        "Select tool",
        ["📝 Rewrite Summary", "🎯 Tailor Section for JD", "💬 Single Bullet Rewrite"],
        horizontal=True,
    )

    st.markdown("---")

    if tool == "📝 Rewrite Summary":
        st.markdown("#### Professional Summary Generator")
        existing = cv.sections.get("summary", None)
        existing_text = existing.content if existing else ""
        if existing_text:
            st.markdown("**Current summary:**")
            st.info(existing_text[:500])

        if st.button("✨ Generate / Improve Summary", type="primary"):
            with st.spinner("Generating summary with Gemini..."):
                st.session_state.summary_draft = generate_summary(
                    cv.full_text, jd, existing_text
                )

        if st.session_state.summary_draft:
            st.markdown("**AI-Generated Summary:**")
            edited_summary = st.text_area(
                "Edit if needed",
                value=st.session_state.summary_draft,
                height=120,
                label_visibility="collapsed",
            )
            st.session_state.summary_draft = edited_summary
            st.success("✅ This will be used in your exported PDF")

    elif tool == "🎯 Tailor Section for JD":
        st.markdown("#### Tailor a Section for the Job Description")
        if not jd:
            st.warning("Paste a job description in the sidebar first.")
        else:
            section_choice = st.selectbox(
                "Which section?",
                options=list(cv.sections.keys()),
                format_func=lambda x: x.title(),
            )
            if st.button("🎯 Tailor This Section", type="primary"):
                sec = cv.sections[section_choice]
                with st.spinner(f"Tailoring {section_choice} section..."):
                    tailored = tailor_cv_for_jd(sec.content, section_choice, jd)

                col_o, col_t = st.columns(2)
                with col_o:
                    st.markdown("**Original:**")
                    st.markdown(f"<div style='background:#1C1E2E;padding:14px;border-radius:8px;font-size:13px;'>{sec.content[:800]}</div>", unsafe_allow_html=True)
                with col_t:
                    st.markdown("**Tailored:**")
                    st.markdown(f"<div style='background:#162016;padding:14px;border-radius:8px;font-size:13px;color:#86EFAC;'>{tailored[:800]}</div>", unsafe_allow_html=True)

    elif tool == "💬 Single Bullet Rewrite":
        st.markdown("#### Rewrite a Single Bullet Point")
        user_bullet = st.text_area(
            "Paste your bullet point",
            height=80,
            placeholder="e.g. Worked on backend APIs for the company's main product",
        )
        sec_name = st.text_input("Section name", value="experience")
        tone_s = st.selectbox("Tone", ["professional", "technical", "concise", "impactful"], key="single_tone")

        if st.button("✨ Rewrite", type="primary") and user_bullet:
            with st.spinner("Rewriting..."):
                result = rewrite_bullet(user_bullet, sec_name, jd, tone_s)
            st.markdown("**Original:**")
            st.markdown(f"<div class='original-text'>{user_bullet}</div>", unsafe_allow_html=True)
            st.markdown("**Rewritten:**")
            st.markdown(f"<div class='rewritten-text'>{result}</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: EXPORT
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("### 📄 Export Optimised CV")

    # Summary of accepted changes
    total_rewrites = sum(
        sum(1 for r in rw if r["accepted"])
        for rw in st.session_state.rewrites.values()
    )
    total_bullets = sum(
        len(rw) for rw in st.session_state.rewrites.values()
    )

    col_s1, col_s2, col_s3 = st.columns(3)
    col_s1.metric("Accepted Rewrites", total_rewrites)
    col_s2.metric("Total Suggestions", total_bullets)
    col_s3.metric("Sections Edited", len(st.session_state.rewrites))

    st.markdown("---")

    # Name / contact override
    st.markdown("#### ✏️ Confirm Contact Details")
    col_n, col_e = st.columns(2)
    with col_n:
        export_name = st.text_input("Full Name", value=contact.get("author") or cv.metadata.get("author") or "")
    with col_e:
        export_email = st.text_input("Email", value=contact.get("email", ""))

    col_p, col_l = st.columns(2)
    with col_p:
        export_phone = st.text_input("Phone", value=contact.get("phone", ""))
    with col_l:
        export_linkedin = st.text_input("LinkedIn", value=contact.get("linkedin", ""))

    # Include summary?
    if st.session_state.summary_draft:
        include_summary = st.checkbox("Include AI-generated summary", value=True)
    else:
        include_summary = False

    st.markdown("---")
    st.markdown("#### 📋 Export Preview")
    st.markdown("The following sections will be included:")
    for sec_name, section in cv.sections.items():
        rw = st.session_state.rewrites.get(sec_name, [])
        accepted = sum(1 for r in rw if r["accepted"])
        total_b = len(rw)
        if total_b > 0:
            st.markdown(f"- **{sec_name.title()}** — {accepted}/{total_b} bullets rewritten")
        else:
            st.markdown(f"- **{sec_name.title()}**")

    st.markdown("---")

    if st.button("⬇️ Generate & Download Optimised CV (PDF)", type="primary", use_container_width=True):
        # Build sections dict for export
        export_sections = {}

        # Override summary if AI draft exists
        if include_summary and st.session_state.summary_draft:
            export_sections["summary"] = st.session_state.summary_draft

        for sec_name, section in cv.sections.items():
            if sec_name == "summary" and include_summary and st.session_state.summary_draft:
                continue
            export_sections[sec_name] = section.content

        export_contact = {
            "email": export_email,
            "phone": export_phone,
            "linkedin": export_linkedin,
        }

        with st.spinner("Generating PDF..."):
            pdf_bytes = export_cv_to_pdf(
                name=export_name,
                contact_info=export_contact,
                sections=export_sections,
                accepted_rewrites=st.session_state.rewrites,
            )

        st.download_button(
            label="📥 Download Optimised CV.pdf",
            data=pdf_bytes,
            file_name="cv_optimised.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
        st.success("✅ PDF ready! Click above to download.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 📊 Export as JSON (raw data)")
    if st.button("Export analysis data as JSON"):
        import json
        data = {
            "ats_score": {
                "total": st.session_state.ats_score.total if st.session_state.ats_score else None,
                "breakdown": {k: list(v) for k, v in (st.session_state.ats_score.breakdown.items() if st.session_state.ats_score else {})},
            },
            "jd_match": {
                "overall": st.session_state.jd_match.overall_score if st.session_state.jd_match else None,
            },
            "rewrites": st.session_state.rewrites,
        }
        st.download_button(
            "📥 Download analysis.json",
            data=json.dumps(data, indent=2),
            file_name="cv_analysis.json",
            mime="application/json",
        )
