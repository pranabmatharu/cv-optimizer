# 📄 CV Optimizer

AI-powered CV analysis, ATS scoring, bullet rewriting, and export — built with Streamlit + Gemini 2.5 Flash.

## Features

| Tab | What it does |
|-----|-------------|
| **ATS Score** | 100-point compliance score across 5 dimensions |
| **JD Match** | TF-IDF cosine similarity + skills gap vs job description |
| **Rewriter** | Gemini rewrites every bullet into impact-driven language |
| **AI Tools** | CV review, summary generator, cover letter, interview Q predictor |
| **Export** | Optimised PDF (with accepted rewrites applied) + plain text + ATS report |

## Setup

### 1. Get a Gemini API key
Go to [aistudio.google.com](https://aistudio.google.com) → **Get API key** (free tier is generous).

### 2. Local development

```bash
git clone https://github.com/YOUR_USERNAME/cv-optimizer.git
cd cv-optimizer
pip install -r requirements.txt

# Add your key to .streamlit/secrets.toml (already gitignored)
echo 'GEMINI_API_KEY = "your-key-here"' > .streamlit/secrets.toml

streamlit run app.py
```

### 3. Deploy on Streamlit Cloud

1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Pick your repo → `app.py` as the main file
4. **Advanced settings → Secrets**, paste:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
5. Deploy — live in ~2 minutes

## Project Structure

```
cv-optimizer/
├── app.py                  # Main Streamlit UI (5 tabs)
├── requirements.txt
├── .gitignore
├── .streamlit/
│   └── secrets.toml        # API keys (gitignored)
└── utils/
    ├── __init__.py         # Makes utils a Python package
    ├── cv_parser.py        # PDF extraction + section segmentation
    ├── ats_scorer.py       # 100-pt ATS engine (5 dimensions)
    ├── jd_matcher.py       # TF-IDF cosine similarity + skills gap
    ├── gemini_client.py    # All Gemini 2.5 Flash API calls
    └── pdf_exporter.py     # ReportLab PDF generator
```

## ATS Score Dimensions

| Dimension | Max | What it checks |
|-----------|-----|----------------|
| Structure | 20 | Required sections present |
| Contact Info | 10 | Email, phone, LinkedIn, GitHub |
| Content Quality | 30 | Bullets, metrics, power verbs, active voice |
| Formatting | 20 | Length, date consistency, ATS-friendly layout |
| Keywords | 20 | Skills density, technical terms, summary strength |

## Tech Stack

- **Streamlit** — UI framework
- **Gemini 2.5 Flash** — All AI features (via `google-generativeai`)
- **pdfplumber / PyPDF2** — PDF text extraction
- **ReportLab** — PDF generation
- **Pure Python** — TF-IDF matching (no scikit-learn dependency)
