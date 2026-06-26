# 📄 Resume Screener AI

An end-to-end AI-powered resume screening tool that matches resumes against job descriptions using a combination of classical machine learning, NLP, and LLM-powered reasoning.

Paste in a job description, upload a resume (or a batch of resumes), and get back a match score, a feature-level explanation of *why* that score was given, and detailed AI-generated feedback — including specific missing skills, ATS keyword gaps, and a rewritten resume bullet point tailored to the role.

## Why this exists

Most "resume matcher" projects either (a) do naive keyword counting, which misses synonyms and context, or (b) just wrap an LLM prompt with no real ML behind it. This project does both — a trained classifier provides a fast, explainable baseline score, and an LLM layer adds the nuanced reasoning a keyword model can't.

## Features

- **Single resume matching** — upload one PDF resume + paste a job description → get a score, SHAP-based explanation, and AI feedback
- **Bulk screening** — upload multiple resumes at once, get a ranked leaderboard, then drill into any individual candidate for full AI analysis
- **PDF resume parsing** — extracts name, contact info, education, GPA, and skills directly from the uploaded file
- **LLM-powered job description parsing** — automatically extracts required skills, programming languages, frameworks, seniority level, and experience requirements from unstructured job postings (via Groq / Llama 3.3-70B)
- **Explainable scoring** — SHAP values show exactly which features drove the prediction up or down
- **AI Recruiter Analysis** — a written match summary, strengths, missing skills with reasoning, improvement tips, a before/after resume bullet rewrite, and a hire recommendation
- **ATS keyword gap detection** — flags keywords from the job description that are missing from the resume

## How it works

1. **Resume & JD parsing** — `pdf_parser.py` extracts structured data from the uploaded PDF; `jd_parser.py` uses an LLM to parse the job description into required skills, seniority, experience, and education requirements (falls back to regex/keyword matching if the LLM call fails)
2. **Feature engineering** (`matcher.py`) — builds an 8-feature vector per resume/JD pair:
   - Multi-label domain alignment (resume's professional domain vs. job's domain, computed as a weighted overlap across 6 categories)
   - Sentence-transformer cosine similarity (full text, and skills-specific)
   - Skill overlap count and percentage
   - Certification presence
   - Education level match
   - Experience level match
3. **Prediction** — a Random Forest classifier (trained on 9,300+ labeled resume-job pairs) outputs a match probability
4. **Explanation** — SHAP TreeExplainer shows each feature's contribution to that specific prediction
5. **AI analysis** — the resume text, JD, and score are passed to an LLM (Groq) which generates the qualitative feedback (strengths, gaps, bullet rewrite, recommendation)

## Tech stack

| Layer | Tools |
|---|---|
| ML model | scikit-learn (Random Forest, Logistic Regression) |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Explainability | SHAP |
| NLP preprocessing | spaCy |
| LLM | Groq API (Llama 3.3-70B) |
| PDF parsing | PyMuPDF |
| App / UI | Streamlit |

## Project structure

```
resume-screener/
├── app.py                      # Streamlit application
├── src/
│   ├── pdf_parser.py           # Resume PDF → structured data
│   ├── jd_parser.py            # Job description → structured data (LLM + fallback)
│   ├── matcher.py              # Shared feature-engineering logic
│   └── llm_analyzer.py         # LLM-powered detailed feedback generation
├── notebooks/
│   ├── eda.ipynb               # Exploratory data analysis
│   ├── feature_engineering.ipynb
│   └── modeling.ipynb          # Model training, evaluation, SHAP
├── models/                     # Trained model, scaler, SHAP explainer (.pkl)
├── data/                       # Processed dataset
└── requirements.txt
```

## Model performance

Trained on 9,320 labeled resume-job pairs (9,300+ after cleaning), with an 8-feature set engineered from domain alignment, semantic similarity, skill overlap, education, and experience matching.

| Model | Accuracy | ROC-AUC |
|---|---|---|
| Logistic Regression | 68.3% | 0.737 |
| **Random Forest** | **69.1%** | **0.754** |

Top features by SHAP importance: domain alignment, full-text semantic similarity, and skills semantic similarity together account for the majority of the model's predictive power.

## Key data insight

During EDA, the original dataset's `matched_score` label showed very weak correlation (< 0.15) with any text-based feature — including exact semantic similarity between resume and job skills. Investigation revealed the score appeared to be driven primarily by structured category matching rather than textual content. Rather than train on a label the model couldn't meaningfully learn from, a custom scoring function was engineered from first principles (domain alignment + skill overlap + semantic similarity + education/experience matching), validated against the original labels, and used to derive better-calibrated binary fit labels for classification.

## Setup

```bash
git clone https://github.com/Abdurahim0202/Resume-Screener-AI.git
cd Resume-Screener-AI
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```

Run the app:
```bash
streamlit run app.py
```

## Future improvements

- Fine-tune a DistilBERT model for direct resume-JD classification
- Bigram/fuzzy skill matching to catch synonyms missed by exact matching
- Resume feedback export as PDF/Word
- Deployment to Streamlit Cloud

## License

MIT
