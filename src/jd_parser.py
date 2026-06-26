from dotenv import load_dotenv
import os
load_dotenv()

import re
import json
from groq import Groq

def parse_jd_with_llm(title, description):
    """
    Use Groq LLM to intelligently extract structured info from any JD format
    """
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    prompt = f"""Extract structured information from this job description and return ONLY a JSON object with no extra text, no markdown, no explanation.

Job Title: {title}

Job Description:
{description}

Return this exact JSON structure:
{{
    "required_skills": ["skill1", "skill2"],
    "preferred_skills": ["skill1", "skill2"],
    "experience_required": null,
    "education_required": null,
    "seniority_level": "internship/junior/mid-level/senior/management",
    "responsibilities": "brief summary",
    "technical_keywords": ["keyword1", "keyword2"],
    "programming_languages": ["python", "javascript"],
    "frameworks_tools": ["langchain", "react"],
    "is_remote": false
}}

Rules:
- Extract ALL programming languages mentioned anywhere in the text
- Extract ALL frameworks, libraries, APIs, and tools mentioned
- Extract ALL required technical and soft skills
- Be thorough, do not leave arrays empty if relevant items exist
- For seniority look for intern, junior, senior, enrolled student keywords
- Return ONLY the JSON object, nothing else"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=1000
    )

    response_text = response.choices[0].message.content.strip()
    response_text = re.sub(r'```json\n?|\n?```', '', response_text).strip()
    return json.loads(response_text)


def parse_jd_keyword_fallback(title, description):
    """
    Fallback keyword-based parser if LLM fails
    """
    description = description.replace('→', '\n')
    description = description.replace('•', '\n')
    description = description.replace('·', '\n')
    description = description.replace('–', '-')

    text = description.strip()
    text_lower = text.lower()
    lines = [line.strip() for line in text.split('\n') if line.strip()]

    section_map = {
        'requirements': ['requirements', 'required', 'qualifications',
                        'what you need', 'must have'],
        'preferred':    ['preferred', 'nice to have', 'bonus', 'desired'],
        'responsibilities': ['responsibilities', 'what you will do',
                            'duties', 'about the role']
    }

    current_section = 'responsibilities'
    section_content = {s: [] for s in section_map}

    for line in lines:
        line_lower = line.lower()
        matched = None
        for section, headers in section_map.items():
            if any(header in line_lower for header in headers):
                matched = section
                break
        if matched:
            current_section = matched
        else:
            section_content[current_section].append(line)

    requirements_text = ' '.join(section_content['requirements'])
    preferred_text = ' '.join(section_content['preferred'])

    return {
        'title': title,
        'required_skills': extract_skills_from_text(requirements_text + ' ' + title),
        'preferred_skills': extract_skills_from_text(preferred_text),
        'experience_required': extract_experience(text),
        'education_required': extract_education_requirement(text),
        'seniority_level': detect_seniority(text_lower),
        'responsibilities': ' '.join(section_content['responsibilities']),
        'technical_keywords': [],
        'programming_languages': [],
        'frameworks_tools': [],
        'is_remote': 'remote' in text_lower,
        'full_text': description
    }


def parse_job_description(title, description):
    """
    Main JD parser - uses Groq LLM for intelligent extraction
    Falls back to keyword matching if LLM fails
    """
    try:
        llm_result = parse_jd_with_llm(title, description)

        llm_result['title'] = title
        llm_result['full_text'] = description

        llm_result.setdefault('required_skills', [])
        llm_result.setdefault('preferred_skills', [])
        llm_result.setdefault('experience_required', None)
        llm_result.setdefault('education_required', None)
        llm_result.setdefault('seniority_level', 'mid-level')
        llm_result.setdefault('responsibilities', '')
        llm_result.setdefault('technical_keywords', [])
        llm_result.setdefault('programming_languages', [])
        llm_result.setdefault('frameworks_tools', [])
        llm_result.setdefault('is_remote', False)

        return llm_result

    except Exception as e:
        print(f"LLM FAILED WITH ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return parse_jd_keyword_fallback(title, description)


def extract_skills_from_text(text):
    skill_list = [
        'python', 'java', 'javascript', 'c++', 'c#', 'sql', 'matlab',
        'html', 'css', 'typescript', 'scala', 'swift', 'kotlin', 'bash',
        'machine learning', 'deep learning', 'nlp', 'computer vision',
        'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas',
        'numpy', 'transformers', 'bert', 'yolo', 'cnn',
        'langchain', 'autogen', 'n8n', 'zapier', 'openai', 'anthropic',
        'llm', 'gpt', 'large language model', 'agentic',
        'llamaindex', 'pinecone', 'chromadb',
        'data analysis', 'data science', 'tableau', 'powerbi',
        'excel', 'hadoop', 'spark', 'etl',
        'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'git', 'linux',
        'react', 'node.js', 'django', 'flask', 'rest api',
        'leadership', 'communication', 'teamwork', 'agile', 'scrum'
    ]
    found = []
    text_lower = text.lower()
    for skill in skill_list:
        if skill in text_lower:
            found.append(skill)
    return found


def extract_experience(text):
    patterns = [
        r'(\d+)\+?\s*years?\s*of\s*experience',
        r'(\d+)\+?\s*years?\s*experience',
        r'minimum\s*(\d+)\s*years?',
        r'at\s*least\s*(\d+)\s*years?'
    ]
    for pattern in patterns:
        match = re.search(pattern, text.lower())
        if match:
            return int(match.group(1))
    return None


def extract_education_requirement(text):
    education_map = {
        'phd': ['phd', 'doctorate', 'doctoral'],
        'master': ['master', 'msc', 'm.s', 'mba', 'postgraduate'],
        'bachelor': ['bachelor', 'b.s', 'b.a', 'undergraduate', 'degree'],
        'associate': ['associate'],
        'high_school': ['high school', 'diploma', 'ged']
    }
    text_lower = text.lower()
    for level, keywords in education_map.items():
        if any(kw in text_lower for kw in keywords):
            return level
    return None


def detect_seniority(text_lower):
    if any(word in text_lower for word in [
        'intern', 'internship', 'student',
        'currently enrolled', 'undergraduate', 'graduate program'
    ]):
        return 'internship'
    elif any(word in text_lower for word in [
        'junior', 'entry', 'associate', 'assistant', 'entry-level'
    ]):
        return 'junior'
    elif any(word in text_lower for word in [
        'senior', 'lead', 'principal', 'staff'
    ]):
        return 'senior'
    elif any(word in text_lower for word in [
        'manager', 'director', 'head', 'vp', 'chief'
    ]):
        return 'management'
    else:
        return 'mid-level'