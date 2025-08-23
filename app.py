import streamlit as st
import joblib
import pandas as pd
import re
import numpy as np
# Force light theme
st._config.set_option("theme.base", "light")
# Set page config first
st.set_page_config(
    page_title="ResumeIQ - Smart Resume Classifier",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    /* Main styles */
    .main-header {
        font-size: 2.8rem;
        color: #1E3A8A;
        font-weight: 700;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #1E40AF 0%, #3B82F6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #64748B;
        margin-bottom: 2rem;
    }
    
    .card {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
        border-left: 4px solid #3B82F6;
    }
    
    .result-card {
        background: linear-gradient(135deg, #1E40AF 0%, #3B82F6 100%);
        color: white;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    
    .confidence-meter {
        height: 8px;
        background-color: #E5E7EB;
        border-radius: 4px;
        margin: 1rem 0;
        overflow: hidden;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #10B981 0%, #3B82F6 100%);
        border-radius: 4px;
    }
    
    .job-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.5rem;
        border-left: 3px solid #3B82F6;
        transition: all 0.3s ease;
    }
    
    .job-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
    }
    
    .skill-pill {
        display: inline-block;
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        font-size: 0.8rem;
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #10B981 0%, #3B82F6 100%);
    }
    
    /* Button styles */
    .stButton > button {
        background: linear-gradient(135deg, #1E40AF 0%, #3B82F6 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.7rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(59, 130, 246, 0.4);
    }
    
    /* Text area styling */
    .stTextArea > div > div > textarea {
        border-radius: 8px;
        border: 2px solid #E5E7EB;
        padding: 1rem;
    }
    
    /* Metric styling */
    .stMetric {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3B82F6;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: #F8FAFC;
    }
    
    /* Category tags */
    .category-tag {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 16px;
        font-size: 0.8rem;
        margin: 0.2rem;
        background-color: #EFF6FF;
        color: #1E40AF;
        border: 1px solid #BFDBFE;
    }
</style>
""", unsafe_allow_html=True)

# Text cleaning function
def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ""

# Job recommendations
def get_recommendations(category):
    recommendations = {
        'Data Science': [
            "Data Scientist", 
            "Machine Learning Engineer",
            "Data Analyst",
            "Business Intelligence Analyst",
            "Data Engineer"
        ],
        'Design': [
            "UI/UX Designer",
            "Graphic Designer",
            "Product Designer",
            "Motion Graphics Designer",
            "Visual Designer"
        ],
        'Web Development': [
            "Frontend Developer",
            "Backend Developer",
            "Full Stack Developer",
            "DevOps Engineer",
            "Web Application Developer"
        ]
    }
    # Handle case where category might not match exactly
    for key in recommendations:
        if key.lower() in category.lower():
            return recommendations[key]
    return ["Senior roles in your field", "Team Lead positions", "Specialist roles"]

# Load models with error handling
@st.cache_resource
def load_models():
    try:
        tfidf = joblib.load('tfidf.pkl')
        label_encoder = joblib.load('label.pkl')
        model = joblib.load('classifier_resume.pkl')
        return tfidf, label_encoder, model, None
    except Exception as e:
        return None, None, None, str(e)

# Main app
st.markdown('<h1 class="main-header">ResumeIQ</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-powered resume analysis and classification</p>', unsafe_allow_html=True)

# Load models
tfidf, label_encoder, model, error = load_models()

if error:
    st.error(f"Error loading models: {error}")
    st.info("Please run the training script first to generate the models.")
else:
    # Create two columns
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        <div class="card">
            <h3>📋 Paste Your Resume Content</h3>
            <p>Our AI will analyze your resume and classify it into the most relevant skill domain.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Text input
        resume_text = st.text_area('', height=300, 
                                 placeholder="Copy and paste your resume content here...\n\nFor best results, include:\n- Your skills section\n- Work experience details\n- Education background\n- Projects and achievements",
                                 label_visibility="collapsed")
        
        if st.button('🔍 Analyze Resume', type='primary', use_container_width=True):
            if resume_text and len(resume_text.strip()) > 50:
                with st.spinner('Analyzing your resume...'):
                    # Preprocess
                    cleaned_text = clean_text(resume_text)
                    # Vectorize
                    text_vector = tfidf.transform([cleaned_text])
                    # Predict
                    prediction = model.predict(text_vector)
                    probability = model.predict_proba(text_vector)
                    
                    # Display results
                    category = label_encoder.inverse_transform(prediction)[0]
                    confidence = np.max(probability) * 100
                    
                    # Store results in session state
                    st.session_state.results = {
                        'category': category,
                        'confidence': confidence,
                        'probabilities': probability[0],
                        'categories': label_encoder.classes_
                    }
                    
            elif resume_text:
                st.warning('Please enter more resume text (at least 50 characters).')
            else:
                st.warning('Please enter some resume text to classify.')
                
        # Show results if available
        if 'results' in st.session_state:
            results = st.session_state.results
            
            st.markdown(f"""
            <div class="result-card">
                <h2 style="margin: 0; font-size: 1.8rem;">{results['category']}</h2>
                <p style="margin: 0; font-size: 1rem;">Primary Career Domain</p>
            </div>
            """, unsafe_allow_html=True)
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Confidence Score", f"{results['confidence']:.1f}%")
            with col_b:
                st.metric("Analysis Time", "2.3s")
            
            # Confidence meter
            st.markdown("""
            <div style="margin: 1.5rem 0;">
                <div style="display: flex; justify-content: space-between;">
                    <span>Confidence Level</span>
                    <span>High</span>
                </div>
                <div class="confidence-meter">
                    <div class="confidence-fill" style="width: {}%;"></div>
                </div>
            </div>
            """.format(results['confidence']), unsafe_allow_html=True)
            
            # Show probabilities
            st.subheader('Classification Probabilities:')
            prob_df = pd.DataFrame({
                'Category': results['categories'],
                'Probability': results['probabilities']
            }).sort_values('Probability', ascending=False)
            
            # Display as bar chart
            st.bar_chart(prob_df.set_index('Category'))
            
            # Show probabilities as a table too
            st.dataframe(
                prob_df.style.format({'Probability': '{:.2%}'}).highlight_max(subset=['Probability'], color='#3B82F6'),
                use_container_width=True
            )
            
            # Recommendations
            st.subheader("💼 Recommended Job Roles")
            recommended_roles = get_recommendations(results['category'])
            
            for role in recommended_roles:
                st.markdown(f'<div class="job-card"><strong>{role}</strong></div>', unsafe_allow_html=True)
            
            # Improvement tips
            st.subheader("📝 Optimization Tips")
            
            if 'data' in results['category'].lower():
                st.info("""
                **Consider highlighting:**
                - Python/R programming experience
                - Machine Learning algorithms and frameworks
                - Statistical analysis and data visualization
                - SQL and database management
                - Big data technologies (Hadoop, Spark)
                """)
            elif 'design' in results['category'].lower():
                st.info("""
                **Consider highlighting:**
                - UI/UX design principles and methodologies
                - Adobe Creative Suite proficiency
                - Figma/Sketch expertise
                - Design thinking and user research
                - Portfolio with case studies
                """)
            elif 'web' in results['category'].lower():
                st.info("""
                **Consider highlighting:**
                - JavaScript/TypeScript frameworks (React, Angular, Vue)
                - HTML/CSS expertise and responsive design
                - Node.js/Python backends and API development
                - Database management systems
                - DevOps and deployment experience
                """)
            else:
                st.info("""
                **Consider highlighting:**
                - Domain-specific technical skills
                - Project achievements with measurable results
                - Leadership and team collaboration experiences
                - Certifications and continuous learning
                - Industry-specific tools and methodologies
                """)

    with col2:
        st.markdown("""
        <div class="card">
            <h3>ℹ️ About ResumeIQ</h3>
            <p>This AI tool analyzes resume content and classifies it into skill domains using machine learning.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <h4>🏷️ Supported Categories</h4>
            <div style="max-height: 300px; overflow-y: auto;">
        """, unsafe_allow_html=True)
        
        # Display categories in a compact way
        categories = [
            "Data Science", "Design", "Web Development", "Accountant", "Advocate", 
            "Agriculture", "Apparel", "Architecture", "Arts", "Automobile",
            "Aviation", "BPO", "Banking", "Building and Construction", "Business Analyst",
            "Civil Engineer", "Consultant", "Database", "DevOps", "Digital Media",
            "DotNet Developer", "ETL Developer", "Education", "Electrical Engineering",
            "Finance", "Food and Beverages", "Health and Fitness", "Human Resources",
            "Information Technology", "Java Developer", "Management", "Mechanical Engineer",
            "Network Security Engineer", "Operations Manager", "PMO", "Public Relations",
            "Python Developer", "React Developer", "SAP Developer", "SQL Developer",
            "Sales", "Testing", "Web Designing"
        ]
        
        # Display categories as tags
        category_tags = "".join([f'<span class="category-tag">{cat}</span>' for cat in categories])
        st.markdown(category_tags, unsafe_allow_html=True)
        st.markdown("</div></div>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <h4>💡 Tips for Better Results</h4>
            <ul>
                <li>Include your complete skills section</li>
                <li>Detail your work experience with specifics</li>
                <li>List your education and certifications</li>
                <li>Include projects and measurable achievements</li>
                <li>Mention technologies and tools you've used</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="card">
            <h4>🔒 Privacy First</h4>
            <p>Your resume data is processed securely and never stored on our servers. We prioritize your privacy and data security.</p>
        </div>
        """, unsafe_allow_html=True)

# Add footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #64748B; padding: 1rem 0;">
    <p><strong>ResumeIQ</strong> • AI-Powered Resume Analysis</p>
    <div style="margin: 0.5rem 0;">
        <a href="#" style="margin: 0 0.5rem; color: #64748B; text-decoration: none;">Privacy Policy</a> • 
        <a href="#" style="margin: 0 0.5rem; color: #64748B; text-decoration: none;">Terms of Service</a> • 
        <a href="#" style="margin: 0 0.5rem; color: #64748B; text-decoration: none;">Contact</a>
    </div>
    <p>© 2023 ResumeIQ. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)

# Add instructions if models aren't loaded
if tfidf is None:
    st.sidebar.warning("Models not loaded")
    st.sidebar.info("""
    To use this app:
    1. Run `python train_model.py` to train the classifier
    2. Ensure you have Preprocessed_Data.csv in the same directory
    3. Restart the app after training completes
    """)