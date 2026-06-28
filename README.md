# 🚀 CV Optimizer AI

An AI-powered CV optimization system built with Streamlit and Google Gemini 2.5 Flash. Upload your CV, paste a job description, and get real-time ATS scoring, semantic keyword matching, and AI-rewritten bullet points — all with a live side-by-side diff editor and PDF export.

---

## ✨ Features

| Feature | Description |
|---|---|
| **ATS Scoring** | 100-point compliance score across 5 dimensions: keywords, structure, power verbs, quantification, length |
| **JD Matching** | TF-IDF cosine similarity matching between CV sections and job description |
| **Keyword Heatmap** | Visual breakdown of which JD terms appear/are missing in your CV |
| **AI Bullet Rewriter** | Gemini 2.5 Flash rewrites bullets into STAR format with strong action verbs |
| **Live Diff Editor** | Side-by-side accept/reject interface for every AI suggestion |
| **Section Tailoring** | Rewrite any CV section to target a specific job description |
| **Summary Generator** | AI-generated professional summary tailored to the role |
| **PDF Export** | Download a clean, ATS-friendly PDF with all accepted changes applied |

---

## 🏗️ Architecture

```
cv_optimizer/
├── app.py                    # Main Streamlit UI (5 tabs)
├── requirements.txt          # Python dependencies
├── .streamlit/
│   ├── config.toml           # Theme and server config
│   └── secrets.toml          # API keys (never commit this)
└── utils/
    ├── cv_parser.py          # PDF extraction + section segmentation
    ├── ats_scorer.py         # ATS compliance scoring engine
    ├── jd_matcher.py         # TF-IDF semantic JD matching
    ├── gemini_client.py      # All Gemini 2.5 Flash API calls
    └── pdf_exporter.py       # ReportLab PDF generation
```

---

## 🚀 Deploy to Streamlit Cloud (free)

### 1. Fork / clone this repo
```bash
git clone https://github.com/YOUR_USERNAME/cv-optimizer.git
cd cv-optimizer
```

### 2. Push to GitHub
```bash
git add .
git commit -m "Initial commit"
git push origin main
```

### 3. Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub account
3. Select this repo → `app.py` as the main file
4. Under **Advanced settings → Secrets**, add:
   ```toml
   GEMINI_API_KEY = "your-gemini-api-key-here"
   ```
5. Click **Deploy**

### 4. Get your Gemini API key
- Visit [aistudio.google.com](https://aistudio.google.com)
- Click **Get API key** → **Create API key**
- Gemini 2.5 Flash is free with generous rate limits

---

## 💻 Run Locally

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Add your API key
echo 'GEMINI_API_KEY = "your-key-here"' > .streamlit/secrets.toml

# Run
streamlit run app.py
```

---

## 🛠️ Tech Stack

- **Frontend/UI**: Streamlit
- **AI/LLM**: Google Gemini 2.5 Flash (`google-generativeai`)
- **PDF Parsing**: pdfplumber + PyMuPDF (fitz)
- **NLP/Matching**: scikit-learn TF-IDF cosine similarity
- **ATS Analysis**: Custom rule-based scorer (regex + keyword matching)
- **Visualisation**: Plotly
- **PDF Export**: ReportLab

---

## 📄 How to Use

1. **Upload** your CV PDF in the left sidebar
2. **Paste** a job description (optional but recommended)
3. Click **⚡ Analyse CV** to run all analysis modules
4. Navigate the 5 tabs:
   - **Dashboard** — ATS score, warnings, AI analysis
   - **Live Editor** — Rewrite bullets section by section, accept/reject
   - **JD Match** — Keyword heatmap, skills gap, section scores
   - **AI Tools** — Summary generator, section tailoring, single bullet rewrite
   - **Export** — Download your optimised CV as PDF

---

## 📊 ATS Scoring Methodology

| Dimension | Weight | What's Checked |
|---|---|---|
| Keyword Match | 30/100 | JD keywords present in CV |
| CV Structure | 25/100 | Required sections (Experience, Education, Skills) |
| Action Verbs | 20/100 | Strong verbs (led, built, engineered, etc.) |
| Quantified Impact | 15/100 | Numbers, %, $, multipliers in achievements |
| Length | 10/100 | Optimal word count (400-900 words) |

---

## ⚠️ Important Notes

- Never commit `.streamlit/secrets.toml` — it's in `.gitignore`
- The Gemini free tier has rate limits — if you hit them, wait 60 seconds
- PDF parsing quality depends on the CV's formatting (text-based PDFs work best; scanned/image PDFs may not parse well)

---

## 🤝 Contributing

PRs welcome! Ideas for extension:
- Add LinkedIn profile URL scraping
- Multi-CV comparison
- Interview question generator based on CV + JD
- CV version history with diff tracking
- Cover letter generator
