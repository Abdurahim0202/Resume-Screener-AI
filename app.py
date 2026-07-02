from dotenv import load_dotenv
import os
load_dotenv()

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import sys
import re

sys.path.append('src')

from src.pdf_parser import parse_resume
import importlib
import src.jd_parser
importlib.reload(src.jd_parser)
from src.jd_parser import parse_job_description
from src.matcher import build_features

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ─── Load models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    rf_model = joblib.load('models/random_forest.pkl')
    scaler = joblib.load('models/scaler.pkl')
    explainer = joblib.load('models/shap_explainer.pkl')
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    return rf_model, scaler, explainer, sentence_model

rf_model, scaler, explainer, sentence_model = load_models()

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Resume Screener AI",
    page_icon="📄",
    layout="wide"
)

st.markdown("""
<style>
    .stApp {
        background-color: #1A1D29;
    }

    h1, h2, h3 {
        color: #F5F3EE !important;
        font-family: 'Helvetica Neue', sans-serif;
    }

    p, .stMarkdown {
        color: #D8D5CC !important;
    }

    /* Score gauge container */
    .score-gauge {
        display: flex;
        justify-content: center;
        align-items: center;
        margin: 1.5rem 0;
    }

    .gauge-circle {
        width: 180px;
        height: 180px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        flex-direction: column;
        font-family: 'Courier New', monospace;
        border: 3px solid #D4A24C;
        background: radial-gradient(circle, rgba(212,162,76,0.08) 0%, transparent 70%);
    }

    .gauge-score {
        font-size: 42px;
        font-weight: 700;
        color: #D4A24C;
    }

    .gauge-label {
        font-size: 12px;
        color: #8B8980;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-top: 4px;
    }

    /* Cards */
    .custom-card {
        background-color: #232838;
        border: 1px solid #353B4F;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
    }

    .custom-card h4 {
        color: #D4A24C !important;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 12px;
    }

    /* Verdict badges */
    .verdict-strong {
        background-color: rgba(95,168,138,0.15);
        color: #5FA88A;
        border: 1px solid #5FA88A;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 600;
        text-align: center;
    }

    .verdict-partial {
        background-color: rgba(212,162,76,0.15);
        color: #D4A24C;
        border: 1px solid #D4A24C;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 600;
        text-align: center;
    }

    .verdict-poor {
        background-color: rgba(196,87,74,0.15);
        color: #C4574A;
        border: 1px solid #C4574A;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 600;
        text-align: center;
    }

    /* Buttons */
    .stButton button {
        background-color: #D4A24C !important;
        color: #1A1D29 !important;
        font-weight: 600 !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 12px !important;
    }

    .stButton button:hover {
        background-color: #C4914A !important;
    }

    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #232838;
        border-radius: 12px;
        padding: 12px;
    }

    /* Text inputs */
    .stTextInput input, .stTextArea textarea {
        background-color: #232838 !important;
        color: #F5F3EE !important;
        border: 1px solid #353B4F !important;
        border-radius: 8px !important;
    }

    /* Progress bars */
    .stProgress > div > div {
        background-color: #D4A24C !important;
    }

    /* Divider */
    hr {
        border-color: #353B4F !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("📄 Resume Screener AI")
st.markdown("Upload a resume and paste a job description to get an instant match score with AI-powered explanation.")
st.divider()

mode = st.radio(
    "Mode",
    ["Single Resume", "Bulk Screening"],
    horizontal=True
)
st.divider()

# Variables that both branches need to define so later code never crashes
uploaded_file = None
uploaded_files = None
resume_data = None
job_title = None
job_description = None
jd_data = None

if mode == "Single Resume":
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📋 Resume")
        uploaded_file = st.file_uploader("Upload Resume (PDF)", type=['pdf'], key="single_resume_uploader")

        if uploaded_file:
            pdf_bytes = uploaded_file.read()
            resume_data = parse_resume(pdf_bytes)

            st.success("✅ Resume uploaded successfully!")
            st.markdown(f"**Name:** {resume_data['name']}")
            if resume_data['email']:
                st.markdown(f"**Email:** {resume_data['email']}")
            if resume_data['linkedin']:
                st.markdown(f"**LinkedIn:** {resume_data['linkedin']}")
            if resume_data['github']:
                st.markdown(f"**GitHub:** {resume_data['github']}")
            if resume_data['degree']:
                st.markdown(f"**Degree:** {resume_data['degree']}")
            if resume_data['gpa']:
                st.markdown(f"**GPA:** {resume_data['gpa']}")
            if resume_data['skills_keywords']:
                st.markdown("**Skills Found:**")
                st.markdown('_' + ' • '.join(resume_data['skills_keywords']) + '_')

    with col2:
        st.subheader("💼 Job Description")
        job_title = st.text_input("Job Title", placeholder="e.g. Machine Learning Engineer", key="single_job_title")
        job_description = st.text_area(
            "Paste Job Description",
            height=300,
            placeholder="Paste the full job description here...",
            key="single_job_description"
        )

        if job_description and job_title:
            with st.spinner("Parsing job description with AI..."):
                jd_data = parse_job_description(job_title, job_description)

            st.success("✅ Job description parsed!")
            st.markdown(f"**Seniority:** {jd_data['seniority_level'].title()}")
            st.markdown(f"**Experience:** {jd_data['experience_required'] or 'Not specified'} {'years' if jd_data['experience_required'] else ''}")
            st.markdown(f"**Education:** {jd_data['education_required'].title() + chr(39) + 's' if jd_data['education_required'] else 'Not specified'}")
            st.markdown(f"**Remote:** {'Yes' if jd_data.get('is_remote') else 'No/Not specified'}")

            if jd_data.get('programming_languages'):
                st.markdown("**Programming Languages:**")
                st.markdown('_' + ' • '.join(jd_data['programming_languages']) + '_')

            if jd_data.get('frameworks_tools'):
                st.markdown("**Frameworks & Tools:**")
                st.markdown('_' + ' • '.join(jd_data['frameworks_tools']) + '_')

            if jd_data['required_skills']:
                st.markdown("**Required Skills:**")
                st.markdown('_' + ' • '.join(jd_data['required_skills']) + '_')

else:  # Bulk Screening mode
    st.subheader("💼 Job Description")
    job_title = st.text_input("Job Title", placeholder="e.g. Machine Learning Engineer", key="bulk_job_title")
    job_description = st.text_area(
        "Paste Job Description",
        height=200,
        placeholder="Paste the full job description here...",
        key="bulk_job_description"
    )

    if job_description and job_title:
        with st.spinner("Parsing job description with AI..."):
            jd_data = parse_job_description(job_title, job_description)
        st.success(f"✅ Parsed! Seniority: {jd_data['seniority_level'].title()}")

    st.subheader("📋 Upload Multiple Resumes")
    uploaded_files = st.file_uploader(
        "Upload Resume PDFs (select multiple)",
        type=['pdf'],
        accept_multiple_files=True,
        key="bulk_resume_uploader"
    )

    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} resumes uploaded")

    if uploaded_files and jd_data:
        st.divider()
        analyze_bulk_btn = st.button("🔍 Screen All Resumes", type="primary", use_container_width=True)

        if analyze_bulk_btn:
            results = []
            st.session_state.bulk_resume_data = {}
            progress_bar = st.progress(0, text="Starting analysis...")

            for i, file in enumerate(uploaded_files):
                pdf_bytes = file.read()
                r_data = parse_resume(pdf_bytes)
                feats, _ = build_features(resume_data, jd_data, None, sentence_model)
                prob = rf_model.predict_proba(feats)[0][1]
                score = round(prob * 100, 1)

                results.append({
                    'Rank': 0,
                    'File': file.name,
                    'Name': r_data['name'],
                    'Score': score,
                    'Degree': r_data['degree'] or 'N/A',
                    'Skills Found': len(r_data['skills_keywords']),
                    'Email': r_data['email'] or 'N/A'
                })

                st.session_state.bulk_resume_data[file.name] = r_data

                progress_bar.progress(
                    (i + 1) / len(uploaded_files),
                    text=f"Analyzed {i+1}/{len(uploaded_files)}: {file.name}"
                )

            progress_bar.empty()

            results_df = pd.DataFrame(results).sort_values('Score', ascending=False).reset_index(drop=True)
            results_df['Rank'] = results_df.index + 1

            st.session_state.bulk_results_df = results_df
            st.session_state.bulk_jd_data = jd_data

        if 'bulk_results_df' in st.session_state:
            results_df = st.session_state.bulk_results_df
            jd_data = st.session_state.bulk_jd_data

            st.subheader("🏆 Candidate Ranking")
            st.dataframe(
                results_df[['Rank', 'File', 'Name', 'Score', 'Degree', 'Skills Found', 'Email']],
                use_container_width=True,
                hide_index=True
            )

            st.success(f"✅ Screened {len(results_df)} candidates. Top match: {results_df.iloc[0]['Name']} ({results_df.iloc[0]['Score']}%)")

            st.divider()
            st.subheader("🔬 Detailed Analysis")
            selected_file = st.selectbox(
                "Select a resume file to see full AI analysis",
                results_df['File'].tolist()
            )

            analyze_selected_btn = st.button("📋 Analyze Selected Candidate", use_container_width=True)

            if analyze_selected_btn:
                selected_resume = st.session_state.bulk_resume_data[selected_file]
                selected_score = results_df[results_df['File'] == selected_file]['Score'].values[0]

                with st.spinner(f"Generating detailed analysis for {selected_file}..."):
                    from src.llm_analyzer import analyze_match_with_llm
                    analysis = analyze_match_with_llm(
                        selected_resume['full_text'],
                        jd_data,
                        selected_score
                    )

                st.markdown('<div class="custom-card">', unsafe_allow_html=True)
                st.markdown(f'<h4>AI Recruiter Analysis — {selected_file}</h4>', unsafe_allow_html=True)

                rec = analysis['hire_recommendation']
                if rec in ('Strong Yes', 'Yes'):
                    st.success(f"✅ Recommendation: {rec} — {analysis['recommendation_reason']}")
                elif rec == 'Maybe':
                    st.warning(f"⚠️ Recommendation: {rec} — {analysis['recommendation_reason']}")
                else:
                    st.error(f"❌ Recommendation: {rec} — {analysis['recommendation_reason']}")

                st.markdown("**📝 Match Summary:**")
                st.write(analysis['match_summary'])

                if analysis.get('missing_skills_reasoning'):
                    st.markdown("**🔎 Why These Gaps Matter:**")
                    st.write(analysis['missing_skills_reasoning'])

                if analysis.get('bullet_rewrite_before') and analysis.get('bullet_rewrite_after'):
                    st.markdown("**✍️ Resume Bullet Rewrite Suggestion:**")
                    st.markdown(f"**Before:** _{analysis['bullet_rewrite_before']}_")
                    st.markdown(f"**After:** {analysis['bullet_rewrite_after']}")

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.markdown("**💪 Strengths:**")
                    for strength in analysis['strengths']:
                        st.markdown(f"✅ {strength}")
                with col_b:
                    st.markdown("**❌ Missing Skills:**")
                    for skill in analysis['missing_skills']:
                        st.markdown(f"❌ {skill}")
                with col_c:
                    st.markdown("**💡 Improvement Tips:**")
                    for tip in analysis['improvement_tips']:
                        st.markdown(f"→ {tip}")

                if analysis['ats_keywords_missing']:
                    st.markdown("**🔍 ATS Keywords Missing from Resume:**")
                    keywords_text = ' • '.join(analysis['ats_keywords_missing'])
                    st.warning(f"Add these to your resume: {keywords_text}")

                st.markdown('</div>', unsafe_allow_html=True)

# ─── Analyze Button (Single Resume mode only) ──────────────────────────────────
if mode == "Single Resume":
    st.divider()
    analyze_btn = st.button("🔍 Analyze Match", type="primary", use_container_width=True)
else:
    analyze_btn = False

if analyze_btn:
    if not uploaded_file or not resume_data:
        st.error("Please upload a resume PDF first!")
    elif not job_description:
        st.error("Please paste a job description!")
    elif not job_title:
        st.error("Please enter a job title!")
    elif not jd_data:
        st.error("Please fill in the job description first!")
    else:
        with st.spinner("Analyzing resume..."):

            # ── Build features (shared logic lives in src/matcher.py) ──
            features, extras = build_features(resume_data, jd_data, nlp, sentence_model)

            domain_alignment   = extras['domain_alignment']
            full_text_sim       = extras['full_text_sim']
            skills_sim           = extras['skills_sim']
            overlap_pct          = extras['overlap_pct']
            has_certification     = extras['has_certification']

            probability = rf_model.predict_proba(features)[0][1]
            score_pct   = round(probability * 100, 1)

            # ── Results ──
            st.divider()
            st.subheader("📊 Results")

            if score_pct >= 70:
                verdict_text = "Strong Match"
                verdict_class = "verdict-strong"
            elif score_pct >= 50:
                verdict_text = "Partial Match"
                verdict_class = "verdict-partial"
            else:
                verdict_text = "Poor Match"
                verdict_class = "verdict-poor"

            col_score, col_verdict = st.columns(2)

            with col_score:
                st.markdown(f"""
                <div class="score-gauge">
                    <div class="gauge-circle">
                        <div class="gauge-score">{score_pct}%</div>
                        <div class="gauge-label">Match Score</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col_verdict:
                st.markdown(f"""
                <div style="display: flex; align-items: center; justify-content: center; height: 180px;">
                    <div class="{verdict_class}" style="width: 100%; font-size: 18px;">
                        {verdict_text}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ── Score breakdown ──
            st.markdown('<div class="custom-card"><h4>Score Breakdown</h4>', unsafe_allow_html=True)
            breakdown = {
                'Domain Alignment':    round(domain_alignment * 100, 1),
                'Semantic Similarity': round(full_text_sim * 100, 1),
                'Skills Similarity':   round(skills_sim * 100, 1),
                'Skill Overlap %':     round(overlap_pct * 100, 1),
                'Has Certification':   has_certification * 100
            }
            for feature, value in breakdown.items():
                st.progress(int(value), text=f"{feature}: {value}%")
            st.markdown('</div>', unsafe_allow_html=True)

            # ── JD Analysis ──
            st.subheader("💼 Job Requirements Analysis")
            jd_col1, jd_col2 = st.columns(2)

            with jd_col1:
                st.markdown(f"**Seniority:** {jd_data['seniority_level'].title()}")
                st.markdown(f"**Experience:** {jd_data['experience_required'] or 'Not specified'} {'years' if jd_data['experience_required'] else ''}")
                st.markdown(f"**Education:** {jd_data['education_required'].title() + chr(39) + 's' if jd_data['education_required'] else 'Not specified'}")
                st.markdown(f"**Remote:** {'Yes' if jd_data.get('is_remote') else 'No/Not specified'}")

                if jd_data.get('programming_languages'):
                    st.markdown("**Programming Languages:**")
                    st.markdown('_' + ' • '.join(jd_data['programming_languages']) + '_')

                if jd_data.get('frameworks_tools'):
                    st.markdown("**Frameworks & Tools:**")
                    st.markdown('_' + ' • '.join(jd_data['frameworks_tools']) + '_')

            with jd_col2:
                resume_skills_lower = [s.lower() for s in resume_data['skills_keywords']]
                if jd_data['required_skills']:
                    st.markdown("**Required Skills Match:**")
                    for skill in jd_data['required_skills']:
                        match = "✅" if skill.lower() in resume_skills_lower else "❌"
                        st.markdown(f"{match} {skill}")

                if jd_data.get('preferred_skills'):
                    st.markdown("**Preferred Skills Match:**")
                    for skill in jd_data['preferred_skills']:
                        match = "✅" if skill.lower() in resume_skills_lower else "❌"
                        st.markdown(f"{match} {skill}")

            # ── SHAP ──
            st.subheader("🧠 AI Explanation (SHAP)")
            shap_values = explainer.shap_values(features)
            fig, ax = plt.subplots(figsize=(10, 4))
            shap_vals = shap_values[0, :, 1] if len(
                np.array(shap_values).shape) == 3 else shap_values[1][0]
            colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in shap_vals]
            ax.barh(features.columns.tolist(), shap_vals, color=colors)
            ax.set_title('Feature Impact\n(Green = increases score, Red = decreases score)')
            ax.set_xlabel('SHAP Value')
            plt.tight_layout()
            st.pyplot(fig)

            # ── LLM Analysis ──
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            st.markdown('<h4>AI Recruiter Analysis</h4>', unsafe_allow_html=True)
            with st.spinner("Generating detailed analysis..."):
                from src.llm_analyzer import analyze_match_with_llm
                analysis = analyze_match_with_llm(
                    resume_data['full_text'],
                    jd_data,
                    score_pct
                )

            rec = analysis['hire_recommendation']
            if rec in ('Strong Yes', 'Yes'):
                st.success(f"✅ Recommendation: {rec} — {analysis['recommendation_reason']}")
            elif rec == 'Maybe':
                st.warning(f"⚠️ Recommendation: {rec} — {analysis['recommendation_reason']}")
            else:
                st.error(f"❌ Recommendation: {rec} — {analysis['recommendation_reason']}")

            st.markdown("**📝 Match Summary:**")
            st.write(analysis['match_summary'])

            if analysis.get('missing_skills_reasoning'):
                st.markdown("**🔎 Why These Gaps Matter:**")
                st.write(analysis['missing_skills_reasoning'])

            if analysis.get('bullet_rewrite_before') and analysis.get('bullet_rewrite_after'):
                st.markdown("**✍️ Resume Bullet Rewrite Suggestion:**")
                st.markdown(f"**Before:** _{analysis['bullet_rewrite_before']}_")
                st.markdown(f"**After:** {analysis['bullet_rewrite_after']}")

            col_a, col_b, col_c = st.columns(3)

            with col_a:
                st.markdown("**💪 Strengths:**")
                for strength in analysis['strengths']:
                    st.markdown(f"✅ {strength}")

            with col_b:
                st.markdown("**❌ Missing Skills:**")
                for skill in analysis['missing_skills']:
                    st.markdown(f"❌ {skill}")

            with col_c:
                st.markdown("**💡 Improvement Tips:**")
                for tip in analysis['improvement_tips']:
                    st.markdown(f"→ {tip}")

            if analysis['ats_keywords_missing']:
                st.markdown("**🔍 ATS Keywords Missing from Resume:**")
                keywords_text = ' • '.join(analysis['ats_keywords_missing'])
                st.warning(f"Add these to your resume: {keywords_text}")

            st.markdown('</div>', unsafe_allow_html=True)