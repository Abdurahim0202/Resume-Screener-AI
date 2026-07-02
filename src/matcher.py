import re
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


STOPWORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be', 'been',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'need', 'this', 'that',
    'these', 'those', 'it', 'its', 'as', 'if', 'up', 'out', 'about',
    'into', 'through', 'during', 'not', 'no', 'so', 'than', 'too', 'very',
    'just', 'also', 'both', 'each', 'more', 'most', 'other', 'such', 'own'
}

def clean_text(text, nlp=None):
    if not isinstance(text, str) or text.strip() == '':
        return ''
    text = re.sub(r"[\[\]'\"{}]", ' ', text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s,]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 1]
    return ' '.join(tokens)


DOMAIN_KEYWORDS = {
    'tech': [
        'python', 'machine learning', 'data science', 'artificial intelligence',
        'deep learning', 'sql', 'java', 'javascript', 'react', 'node',
        'aws', 'cloud', 'docker', 'kubernetes', 'devops', 'software',
        'programming', 'developer', 'engineer', 'database', 'api',
        'tensorflow', 'pytorch', 'keras', 'nlp', 'computer vision',
        'full stack', 'backend', 'frontend', 'ios', 'android', 'mobile',
        'llm', 'langchain', 'openai', 'anthropic', 'gpt', 'agentic'
    ],
    'data': [
        'data analysis', 'data analytics', 'business intelligence',
        'tableau', 'powerbi', 'excel', 'statistics', 'visualization',
        'reporting', 'dashboard', 'data warehouse', 'etl', 'hadoop',
        'spark', 'big data', 'data engineer', 'data science'
    ],
    'engineering': [
        'autocad', 'civil', 'mechanical', 'structural', 'construction',
        'site engineer', 'electrical', 'architecture', 'blueprint',
        'project management', 'manufacturing', 'embedded', 'hardware',
        'microprocessor', 'circuit', 'cad', 'etabs', 'ms project'
    ],
    'finance': [
        'accounting', 'audit', 'vat', 'budget', 'banking', 'finance',
        'accounts payable', 'accounts receivable', 'tax', 'financial',
        'bookkeeping', 'compliance', 'internal control', 'treasury',
        'quickbooks', 'tally', 'ifrs', 'gaap'
    ],
    'marketing': [
        'marketing', 'sales', 'brand', 'campaign', 'trade marketing',
        'business development', 'customer', 'market research',
        'digital marketing', 'seo', 'social media', 'advertising',
        'merchandising', 'promotional', 'crm'
    ],
    'admin': [
        'administration', 'hr', 'human resources', 'office management',
        'recruitment', 'payroll', 'operations', 'administrative',
        'secretary', 'coordinator', 'management trainee', 'executive'
    ]
}


def compute_domain_vector(text):
    if not isinstance(text, str) or text.strip() == '':
        return {domain: 0 for domain in DOMAIN_KEYWORDS}
    text = text.lower()
    return {domain: sum(1 for kw in keywords if kw in text)
            for domain, keywords in DOMAIN_KEYWORDS.items()}


def compute_multilabel_alignment(resume_text, job_text):
    rv = compute_domain_vector(resume_text)
    jv = compute_domain_vector(job_text)
    rt = sum(rv.values()) or 1
    jt = sum(jv.values()) or 1
    rn = {d: s / rt for d, s in rv.items()}
    jn = {d: s / jt for d, s in jv.items()}
    return round(sum(rn[d] * jn[d] for d in DOMAIN_KEYWORDS), 4)


def compute_semantic_similarity(text1, text2, sentence_model):
    if not text1 or not text2:
        return 0
    embeddings = sentence_model.encode([text1, text2])
    return float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])


def compute_skill_overlap(resume_skills, required_skills):
    if not required_skills:
        return 0, 0
    resume_set = set([s.lower() for s in resume_skills])
    required_set = set([s.lower() for s in required_skills])
    overlap = resume_set.intersection(required_set)
    return len(overlap), len(overlap) / len(required_set) if required_set else 0


def education_level_score(text):
    if not isinstance(text, str):
        return 0
    text = text.lower()
    if any(w in text for w in ['phd', 'doctorate', 'doctoral']):
        return 4
    elif any(w in text for w in ['master', 'msc', 'm.s', 'mba']):
        return 3
    elif any(w in text for w in ['bachelor', 'b.s', 'b.a', 'undergraduate']):
        return 2
    elif any(w in text for w in ['associate', 'diploma']):
        return 1
    return 0


def compute_education_match(candidate_degree, required_education):
    candidate_level = education_level_score(candidate_degree)
    required_level = education_level_score(required_education) if required_education else 0
    if required_level == 0:
        return 1.0
    diff = candidate_level - required_level
    if diff >= 0:
        return 1.0
    elif diff == -1:
        return 0.5
    return 0.0


def compute_experience_match(position_count, required_years):
    if required_years is None:
        return 1.0
    try:
        required_years = float(required_years)
    except (ValueError, TypeError):
        return 1.0
    if required_years == 0:
        return 1.0
    estimated_years = position_count * 1.5
    diff = estimated_years - required_years
    if diff >= 0:
        return 1.0
    elif diff >= -2:
        return 0.5
    return 0.0


def build_features(resume_data, jd_data, nlp, sentence_model):
    """
    Given parsed resume_data and jd_data, compute the full 8-feature vector.
    Returns (features_df, intermediate_values_dict)
    """
    resume_text = clean_text(resume_data['full_text'], nlp)
    job_text = clean_text(
        jd_data['title'] + ' ' +
        ' '.join(jd_data['required_skills']) + ' ' +
        ' '.join(jd_data.get('programming_languages', [])) + ' ' +
        ' '.join(jd_data.get('frameworks_tools', [])) + ' ' +
        jd_data['responsibilities'] + ' ' +
        jd_data['full_text'],
        nlp
    )

    domain_alignment = compute_multilabel_alignment(resume_text, job_text)
    full_text_sim = compute_semantic_similarity(resume_text, job_text, sentence_model)
    skills_sim = compute_semantic_similarity(
        ' '.join(resume_data['skills_keywords']),
        ' '.join(jd_data['required_skills'] +
                 jd_data.get('programming_languages', []) +
                 jd_data.get('frameworks_tools', [])),
        sentence_model
    )
    overlap_count, overlap_pct = compute_skill_overlap(
        resume_data['skills_keywords'],
        jd_data['required_skills'] + jd_data.get('programming_languages', [])
    )
    has_certification = 1 if any(
        word in resume_data['full_text'].lower()
        for word in ['certification', 'certified', 'certificate']
    ) else 0

    education_match = compute_education_match(
        resume_data.get('degree', ''),
        jd_data.get('education_required', '')
    )

    position_count = len(re.findall(
        r'\b(intern|developer|engineer|analyst|assistant|manager)\b',
        resume_data['full_text'].lower()
    ))
    position_count = max(position_count, 1)

    experience_match = compute_experience_match(
        position_count,
        jd_data.get('experience_required')
    )

    features = pd.DataFrame([[
        domain_alignment, overlap_count, overlap_pct, full_text_sim,
        skills_sim, has_certification, education_match, experience_match
    ]], columns=[
        'domain_alignment_v2', 'skill_overlap_count', 'skill_overlap_pct',
        'full_text_semantic_sim', 'skills_semantic_sim', 'has_certification',
        'education_match', 'experience_match'
    ])

    extras = {
        'domain_alignment': domain_alignment,
        'full_text_sim': full_text_sim,
        'skills_sim': skills_sim,
        'overlap_pct': overlap_pct,
        'has_certification': has_certification
    }

    return features, extras