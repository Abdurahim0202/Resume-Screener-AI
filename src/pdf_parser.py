import fitz  # PyMuPDF
import re

def extract_text_from_pdf(pdf_bytes):
    """Extract raw text from PDF bytes (from Streamlit uploader)"""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    return full_text

def parse_resume(pdf_bytes):
    """
    Main function - parses resume PDF into structured sections
    Designed for standard resume format with clear section headers
    """
    raw_text = extract_text_from_pdf(pdf_bytes)
    lines = raw_text.split('\n')
    lines = [line.strip() for line in lines if line.strip() != '']

    # Define section headers
    section_map = {
        'education':  ['education'],
        'experience': ['experience', 'work experience', 'employment'],
        'projects':   ['projects', 'personal projects'],
        'skills':     ['skills', 'technical skills', 'core competencies'],
        'objective':  ['summary', 'objective', 'profile', 'about']
    }

    sections = {s: [] for s in section_map}
    sections['header'] = []
    current_section = 'header'

    for line in lines:
        line_lower = line.lower()

        # Check if line is a section header
        matched_section = None
        for section, headers in section_map.items():
            if any(line_lower == h or line_lower.startswith(h) for h in headers):
                matched_section = section
                break

        if matched_section:
            current_section = matched_section
        else:
            sections[current_section].append(line)

    # Join sections into strings
    result = {s: ' '.join(sections[s]) for s in sections}

    # Extract contact info from header
    header_text = ' '.join(sections['header'])
    result['name'] = lines[0] if lines else 'Unknown'
    result['email'] = extract_email(header_text)
    result['linkedin'] = extract_linkedin(header_text)
    result['github'] = extract_github(header_text)

    # Extract skills from both skills section and full text
    result['skills_keywords'] = extract_skills_keywords(
        result['skills'] + ' ' + result['experience'] + ' ' + result['projects']
    )

    # Extract education details
    result['degree'] = extract_degree(result['education'])
    result['gpa'] = extract_gpa(result['education'])

    # Full text for semantic similarity
    result['full_text'] = raw_text

    return result

def extract_email(text):
    match = re.search(r'[\w.-]+@[\w.-]+\.\w+', text)
    return match.group(0) if match else None

def extract_linkedin(text):
    match = re.search(r'linkedin\.com/in/[\w-]+', text)
    return match.group(0) if match else None

def extract_github(text):
    match = re.search(r'github\.com/[\w-]+', text)
    return match.group(0) if match else None

def extract_gpa(text):
    match = re.search(r'GPA[:\s]*([\d.]+)', text, re.IGNORECASE)
    return float(match.group(1)) if match else None

def extract_degree(text):
    degrees = [
        'bachelor', 'master', 'phd', 'doctorate',
        'b.s', 'b.a', 'm.s', 'm.a', 'mba', 'associate'
    ]
    text_lower = text.lower()
    for degree in degrees:
        if degree in text_lower:
            return degree.upper()
    return None

def extract_skills_keywords(text):
    """
    Extract technical skills and keywords from text
    """
    # Technical skills to look for
    skill_list = [
        # Programming languages
        'python', 'java', 'javascript', 'c++', 'c#', 'sql', 'r',
        'html', 'css', 'typescript', 'scala', 'swift', 'kotlin',
        # ML/AI
        'machine learning', 'deep learning', 'nlp', 'computer vision',
        'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas',
        'numpy', 'matplotlib', 'transformers', 'bert',
        # Data
        'data analysis', 'data science', 'tableau', 'powerbi',
        'excel', 'sql', 'hadoop', 'spark', 'etl',
        # Web/Cloud
        'react', 'node.js', 'django', 'flask', 'aws', 'azure',
        'docker', 'kubernetes', 'git', 'linux',
        # Soft skills
        'leadership', 'project management', 'agile', 'scrum',
        'communication', 'teamwork'
    ]

    found_skills = []
    text_lower = text.lower()

    for skill in skill_list:
        if skill in text_lower:
            found_skills.append(skill)

    return found_skills