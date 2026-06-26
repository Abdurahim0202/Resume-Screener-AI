from dotenv import load_dotenv
import os
load_dotenv()
from groq import Groq
import re

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def analyze_match_with_llm(resume_text, jd_data, score_pct):
    """
    Use Groq to generate detailed match analysis
    """
    prompt = f"""You are an expert recruiter and career coach. Analyze this resume against the job description and provide detailed, specific feedback.

Job Title: {jd_data['title']}
Required Skills: {', '.join(jd_data['required_skills'])}
Programming Languages: {', '.join(jd_data.get('programming_languages', []))}
Frameworks: {', '.join(jd_data.get('frameworks_tools', []))}
Seniority: {jd_data['seniority_level']}
Match Score: {score_pct}%

Resume Summary:
{resume_text[:2500]}

Provide a JSON response with this exact structure:
{{
    "match_summary": "3-4 sentence overall assessment explaining the score in detail",
    "strengths": ["strength1", "strength2", "strength3"],
    "missing_skills": ["skill1", "skill2"],
    "missing_skills_reasoning": "2-3 sentences explaining WHY these specific missing skills matter for THIS role, referencing the job's actual responsibilities",
    "improvement_tips": ["tip1", "tip2", "tip3"],
    "bullet_rewrite_before": "pick one weak or generic bullet point from the candidate's resume, quoted close to as-is",
    "bullet_rewrite_after": "a rewritten, stronger version of that bullet point tailored to better match this specific job description, using metrics where plausible",
    "ats_keywords_missing": ["keyword1", "keyword2"],
    "hire_recommendation": "Strong Yes / Yes / Maybe / No",
    "recommendation_reason": "2 sentence reason with specific evidence from the resume"
}}

Be specific and reference actual content from the resume and job description. Avoid generic advice. Return ONLY the JSON object, nothing else."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=1500
    )

    response_text = response.choices[0].message.content.strip()
    response_text = re.sub(r'```json\n?|\n?```', '', response_text).strip()

    import json
    return json.loads(response_text)