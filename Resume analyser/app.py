import re
import joblib
import pdfplumber
import pytesseract
from PIL import Image
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

app = Flask(__name__)

# Load models
try:
    model = joblib.load('resume_classifier_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    label_encoder = joblib.load('label_encoder.pkl')
except Exception as e:
    print(f"Error loading models: {e}")
    model = None
    vectorizer = None
    label_encoder = None

# Extended Skills Database for ATS Scoring
JOB_SKILLS_DB = {
    'Data Science': ['python', 'r', 'sql', 'pandas', 'numpy', 'scikit-learn', 'tensorflow', 'keras', 'pytorch', 'tableau', 'power bi', 'machine learning', 'deep learning', 'statistics', 'big data', 'hadoop', 'spark'],
    'Web Designing': ['html', 'css', 'javascript', 'jquery', 'bootstrap', 'photoshop', 'illustrator', 'figma', 'adobe xd', 'responsive design', 'ui/ux', 'wireframing'],
    'Java Developer': ['java', 'spring', 'spring boot', 'hibernate', 'maven', 'gradle', 'sql', 'jdbc', 'jpa', 'rest api', 'microservices', 'git', 'junit'],
    'Python Developer': ['python', 'django', 'flask', 'fastapi', 'sql', 'postgresql', 'mongodb', 'docker', 'kubernetes', 'aws', 'azure', 'git', 'rest api'],
    'DevOps Engineer': ['linux', 'bash', 'python', 'docker', 'kubernetes', 'jenkins', 'ansible', 'terraform', 'aws', 'azure', 'gcp', 'git', 'ci/cd', 'monitoring'],
    'Network Security Engineer': ['networking', 'tcp/ip', 'firewall', 'vpn', 'ids/ips', 'wireshark', 'linux', 'python', 'bash', 'security', 'penetration testing', 'vulnerability assessment'],
    'HR': ['recruitment', 'employee relations', 'performance management', 'payroll', 'compliance', 'communication', 'negotiation', 'leadership', 'microsoft office', 'hrms'],
    'Sales': ['sales', 'marketing', 'communication', 'negotiation', 'crm', 'lead generation', 'customer service', 'presentation', 'microsoft office', 'b2b', 'b2c'],
    'Mechanical Engineer': ['autocad', 'solidworks', 'catia', 'ansys', 'matlab', 'thermodynamics', 'fluid mechanics', 'manufacturing', 'design', 'project management'],
    'Arts': ['creativity', 'communication', 'design', 'illustration', 'painting', 'sketching', 'adobe creative suite', 'photography', 'visual arts'],
    'Database': ['sql', 'mysql', 'postgresql', 'oracle', 'mongodb', 'cassandra', 'redis', 'database design', 'normalization', 'performance tuning', 'backup', 'recovery'],
    'Electrical Engineering': ['matlab', 'simulink', 'autocad', 'plc', 'scada', 'circuit design', 'power systems', 'electronics', 'control systems', 'instrumentation'],
    'Health and fitness': ['nutrition', 'exercise physiology', 'anatomy', 'kinesiology', 'personal training', 'cpr', 'first aid', 'communication', 'motivation'],
    'PMO': ['project management', 'program management', 'portfolio management', 'agile', 'scrum', 'waterfall', 'jira', 'confluence', 'ms project', 'risk management', 'stakeholder management'],
    'Business Analyst': ['sql', 'excel', 'tableau', 'power bi', 'python', 'r', 'data analysis', 'requirements gathering', 'process mapping', 'agile', 'scrum', 'communication'],
    'Civil Engineer': ['autocad', 'civil 3d', 'staad pro', 'etabs', 'revit', 'project management', 'construction', 'structural analysis', 'surveying', 'estimation'],
    'Automation Testing': ['selenium', 'java', 'python', 'cucumber', 'testng', 'junit', 'jenkins', 'git', 'api testing', 'postman', 'sql', 'agile'],
    'Blockchain': ['blockchain', 'ethereum', 'solidity', 'smart contracts', 'cryptography', 'distributed ledger', 'hyperledger', 'web3', 'dapps', 'consensus algorithms'],
    'Operations Manager': ['operations management', 'supply chain', 'logistics', 'process improvement', 'lean six sigma', 'budgeting', 'leadership', 'communication', 'project management'],
    'SAP Developer': ['sap', 'abap', 'hana', 'fiori', 'sap modules', 'erp', 'integration', 'debugging', 'performance tuning'],
    'ETL Developer': ['etl', 'sql', 'informatica', 'talend', 'ssis', 'data warehousing', 'database', 'python', 'scripting', 'data modeling'],
    'DotNet Developer': ['c#', '.net', 'asp.net', 'mvc', 'web api', 'sql server', 'entity framework', 'linq', 'javascript', 'html', 'css', 'azure'],
    'Hadoop': ['hadoop', 'hdfs', 'mapreduce', 'hive', 'pig', 'hbase', 'spark', 'flume', 'sqoop', 'oozie', 'zookeeper', 'big data'],
    'Testing': ['manual testing', 'automation testing', 'selenium', 'jira', 'bug tracking', 'test cases', 'test plans', 'sql', 'agile', 'sdlc']
}

def clean_resume(text):
    """
    Clean the resume text by removing URLs, emails, special characters,
    numbers, extra spaces, and stopwords.
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove emails
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters and numbers (keep only letters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove stopwords
    words = text.split()
    filtered_words = [word for word in words if word not in ENGLISH_STOP_WORDS]
    
    return " ".join(filtered_words)

def extract_text_from_pdf(file_stream):
    text = ""
    try:
        with pdfplumber.open(file_stream) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
    return text

def extract_text_from_image(file_stream):
    text = ""
    try:
        image = Image.open(file_stream)
        text = pytesseract.image_to_string(image)
    except Exception as e:
        print(f"Error extracting Image text: {e}")
    return text

def calculate_ats_score(text, predicted_category):
    """
    Calculate ATS score (0-100) based on skills, sections, keyword density, and length.
    """
    score = 0
    text_lower = text.lower()
    words = text_lower.split()
    word_count = len(words)
    
    # 1. Skill Match (40 points)
    category_skills = JOB_SKILLS_DB.get(predicted_category, [])
    if category_skills:
        matched_skills = [skill for skill in category_skills if skill in text_lower]
        skill_match_ratio = len(matched_skills) / len(category_skills) if len(category_skills) > 0 else 0
        score += min(40, skill_match_ratio * 100 * 0.8) # Cap at 40, heuristic multiplier
        # Bonus for having at least some skills
        if len(matched_skills) > 0:
             score = max(score, 10) # Minimum 10 points if any skill matches

    # 2. Resume Sections (30 points)
    sections = ['education', 'experience', 'skills', 'projects', 'certifications', 'summary', 'objective']
    found_sections = [sec for sec in sections if sec in text_lower]
    section_score = (len(found_sections) / 5) * 30 # Expecting at least 5 key sections
    score += min(30, section_score)

    # 3. Keyword Density (20 points)
    # Simple heuristic: if we found skills, how frequently do they appear?
    # Or just general length of relevant content.
    # Let's use a simple check: if we have matched skills, give points.
    if category_skills:
        matched_count = sum(text_lower.count(skill) for skill in category_skills)
        if matched_count > 10:
            score += 20
        elif matched_count > 5:
            score += 10
        elif matched_count > 0:
            score += 5
    else:
        # Fallback if category not in DB
        score += 10

    # 4. Resume Length & Structure (10 points)
    if word_count > 200:
        score += 10
    elif word_count > 100:
        score += 5
    
    return min(100, round(score))

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    filename = None
    ats_score = 0
    
    if request.method == 'POST':
        if 'resume_file' not in request.files:
            return render_template('index.html', error="No file part")
        
        file = request.files['resume_file']
        
        if file.filename == '':
            return render_template('index.html', error="No selected file")
        
        if file:
            filename = file.filename
            text = ""
            
            if filename.lower().endswith('.pdf'):
                text = extract_text_from_pdf(file)
            elif filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                text = extract_text_from_image(file)
            else:
                return render_template('index.html', error="Invalid file type")
            
            if text and model and vectorizer and label_encoder:
                cleaned_text = clean_resume(text)
                vectorized_text = vectorizer.transform([cleaned_text])
                predicted_label = model.predict(vectorized_text)
                prediction = label_encoder.inverse_transform(predicted_label)[0]
                
                # Calculate ATS Score
                ats_score = calculate_ats_score(text, prediction)
            
    return render_template('index.html', prediction=prediction, filename=filename, ats_score=ats_score)

if __name__ == '__main__':
    app.run(debug=True)
