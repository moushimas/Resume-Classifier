import streamlit as st
import pdfplumber
import docx2txt
import joblib
import nltk
import re
import string

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# -------------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------------

st.set_page_config(
    page_title="🤖 AI Resume Classifier",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------------

st.markdown("""
<style>

.stApp{
background:linear-gradient(135deg,#f0f4ff,#ffffff,#eef6ff);
}

.block-container{
padding-top:2rem;
padding-bottom:2rem;
}

section[data-testid="stSidebar"]{
background:#0F172A;
}

section[data-testid="stSidebar"] *{
color:white;
}

h1,h2,h3{
font-family:Arial;
font-weight:bold;
}

.metric-card{
background:white;
padding:20px;
border-radius:15px;
box-shadow:0px 6px 15px rgba(0,0,0,.15);
text-align:center;
}

.result-card{
background:linear-gradient(135deg,#2563EB,#0EA5E9);
padding:30px;
border-radius:18px;
color:white;
text-align:center;
box-shadow:0px 8px 20px rgba(0,0,0,.2);
}

.upload-card{
background:white;
padding:20px;
border-radius:15px;
box-shadow:0px 5px 15px rgba(0,0,0,.1);
}

.stButton>button{
background:#2563EB;
color:white;
border:none;
border-radius:10px;
font-size:18px;
font-weight:bold;
height:50px;
width:100%;
}

.stButton>button:hover{
background:#1D4ED8;
color:white;
}

div[data-testid="stMetric"]{
background:white;
padding:15px;
border-radius:12px;
box-shadow:0px 5px 10px rgba(0,0,0,.1);
}

</style>
""",unsafe_allow_html=True)

# -------------------------------------------------------
# NLTK
# -------------------------------------------------------

try:
    stop_words=set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords",quiet=True)
    stop_words=set(stopwords.words("english"))

stemmer=PorterStemmer()

# -------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------

@st.cache_resource
def load_model():

    model=joblib.load("resume_classifier_model.pkl")
    vectorizer=joblib.load("tfidf_vectorizer.pkl")

    return model,vectorizer

try:
    model,vectorizer=load_model()

except Exception as e:

    st.error(f"Model Loading Error : {e}")
    st.stop()

# -------------------------------------------------------
# BEAUTIFUL SIDEBAR
# -------------------------------------------------------

with st.sidebar:

    st.markdown("# 🤖 Resume AI")

    st.image(
        "https://img.icons8.com/color/240/artificial-intelligence.png",
        width=120
    )

    st.success("Machine Learning Project")

    st.markdown("---")

    st.markdown("## 📌 About")

    st.write("""
This application classifies resumes automatically using Machine Learning and NLP.
""")

    st.markdown("---")

    st.markdown("## 🤖 Model")

    st.info("""
Algorithm : Linear SVM

Feature Extraction : TF-IDF

Resume Categories : 4
""")

    st.markdown("---")

    st.markdown("## 📂 Supported Files")

    st.write("✅ PDF")

    st.write("✅ DOCX")

# -------------------------------------------------------
# MAIN HEADER
# -------------------------------------------------------

st.markdown("""
<h1 style='text-align:center;
color:#1E3A8A;
font-size:48px;
font-weight:bold;'>

🤖 AI Resume Classification System

</h1>

<h4 style='text-align:center;
color:#475569;'>

Smart Resume Analysis Using Machine Learning & NLP

</h4>

""",unsafe_allow_html=True)

st.markdown("---")

col1,col2,col3=st.columns(3)

with col1:
    st.metric("🤖 Model","Linear SVM")

with col2:
    st.metric("📄 Categories","4")

with col3:
    st.metric("🧠 NLP","TF-IDF")
    # ============================================================
# 11. UPLOAD RESUME
# ============================================================

st.markdown("""
<div class="upload-card">

<h2 style="color:#2563EB;">
📤 Upload Your Resume
</h2>

<p style="font-size:17px;">
Upload your resume in PDF or DOCX format.
The AI model will analyze your resume and
predict the most suitable job category.
</p>

</div>

""",unsafe_allow_html=True)

st.write("")

uploaded_file = st.file_uploader(
    "📂 Choose Resume",
    type=["pdf","docx"]
)

st.write("")

if uploaded_file is not None:

    st.success(f"✅ {uploaded_file.name} uploaded successfully")

    try:

        # --------------------------------------------
        # Extract Resume
        # --------------------------------------------

        if uploaded_file.name.lower().endswith(".pdf"):

            with st.spinner("📖 Reading PDF Resume..."):

                resume_text = extract_pdf(uploaded_file)

        elif uploaded_file.name.lower().endswith(".docx"):

            with st.spinner("📖 Reading DOCX Resume..."):

                resume_text = extract_docx(uploaded_file)

        else:

            st.error("Unsupported File")

            st.stop()

        # --------------------------------------------
        # Validate Resume
        # --------------------------------------------

        if not resume_text.strip():

            st.error("Resume text not found.")

            st.stop()

        st.markdown("---")

        st.markdown("""
        <h2 style="color:#1E40AF;">
        📊 Resume Statistics
        </h2>
        """,unsafe_allow_html=True)

        c1,c2,c3,c4 = st.columns(4)

        with c1:
            st.metric(
                "📄 File Type",
                uploaded_file.name.split(".")[-1].upper()
            )

        with c2:
            st.metric(
                "📝 Words",
                len(resume_text.split())
            )

        with c3:
            st.metric(
                "🔤 Characters",
                len(resume_text)
            )

        with c4:
            st.metric(
                "💾 File Size",
                f"{uploaded_file.size/1024:.1f} KB"
            )

        st.write("")

        st.markdown("""
        <h2 style="color:#2563EB;">
        👀 Resume Preview
        </h2>
        """,unsafe_allow_html=True)

        with st.expander("Click to View Resume"):

            st.text_area(
                "",
                value=resume_text[:5000],
                height=350,
                disabled=True
            )

        st.write("")

        with st.spinner("🧹 Cleaning Resume..."):

            clean_resume = preprocess_text(resume_text)

        if not clean_resume.strip():

            st.error("Resume does not contain enough useful information.")

            st.stop()
                    # ==========================================================
        # TF-IDF TRANSFORMATION
        # ==========================================================

        with st.spinner("🧠 Extracting Features..."):

            resume_vector = vectorizer.transform(
                [clean_resume]
            )

        # ==========================================================
        # AI PREDICTION
        # ==========================================================

        with st.spinner("🤖 AI is Predicting..."):

            prediction = model.predict(
                resume_vector
            )[0]

        st.markdown("---")

        st.markdown(
            """
            <h2 style="text-align:center;
            color:#1E3A8A;">
            🎯 Prediction Result
            </h2>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="result-card">

            <h1>
            {prediction}
            </h1>

            <p style="font-size:20px;">
            Resume Successfully Classified
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("")

        left,right=st.columns([2,1])

        with left:

            st.success(
                "The AI model analyzed the resume "
                "and predicted the most suitable category."
            )

            st.write("""
The prediction is based on:

• Technical Skills

• Programming Languages

• Experience

• Technologies

• Keywords extracted from the resume.
""")

        with right:

            st.metric(
                "🏆 Predicted Role",
                prediction
            )

        st.info(
            """
The model predicts only one of the
following categories:

✅ React Developer

✅ SQL Developer

✅ PeopleSoft

✅ Workday
"""
        )

    except Exception as e:

        st.error("❌ Something went wrong.")

        st.exception(e)
        # ============================================================
# HOW THE SYSTEM WORKS
# ============================================================

st.markdown("---")

st.markdown("""
<h2 style="text-align:center;color:#1E3A8A;">
⚙️ How The System Works
</h2>
""", unsafe_allow_html=True)

c1,c2,c3,c4=st.columns(4)

with c1:
    st.markdown("""
    <div class="metric-card">
    <h3>📤 Upload</h3>
    <p>Select a PDF or DOCX Resume</p>
    </div>
    """,unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="metric-card">
    <h3>🧹 Clean</h3>
    <p>Remove stopwords, punctuation and unwanted text.</p>
    </div>
    """,unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="metric-card">
    <h3>🧠 TF-IDF</h3>
    <p>Convert text into numerical vectors.</p>
    </div>
    """,unsafe_allow_html=True)

with c4:
    st.markdown("""
    <div class="metric-card">
    <h3>🤖 Predict</h3>
    <p>Linear SVM predicts the resume category.</p>
    </div>
    """,unsafe_allow_html=True)

# ============================================================
# TECHNOLOGIES
# ============================================================

st.markdown("---")

st.markdown("""
<h2 style="text-align:center;color:#1E3A8A;">
🛠 Technologies Used
</h2>
""",unsafe_allow_html=True)

col1,col2,col3=st.columns(3)

with col1:
    st.info("""
### 🐍 Python

Application Development
""")

with col2:
    st.success("""
### 🧠 Machine Learning

Linear SVM

TF-IDF

NLP
""")

with col3:
    st.warning("""
### 🌐 Streamlit

Interactive Web Application
""")

# ============================================================
# PROJECT INFORMATION
# ============================================================

st.markdown("---")

st.markdown("""
<div style="
background:white;
padding:25px;
border-radius:15px;
box-shadow:0px 5px 15px rgba(0,0,0,.1);
">

<h2 style="color:#2563EB;">
📌 Project Information
</h2>

<b>Project :</b> AI Resume Classification System<br><br>

<b>Algorithm :</b> Linear Support Vector Machine<br><br>

<b>Feature Extraction :</b> TF-IDF<br><br>

<b>Natural Language Processing :</b>
Text Cleaning,
Stopword Removal,
Stemming<br><br>

<b>Framework :</b> Streamlit

</div>

""",unsafe_allow_html=True)

# ============================================================
# FOOTER
# ============================================================

st.markdown("<br><br>",unsafe_allow_html=True)

st.markdown("""
<hr>

<center>

<h2 style="color:#1E3A8A;">
🤖 AI Resume Classification System
</h2>

<h4>
Machine Learning • NLP • Streamlit
</h4>

<p style="color:gray;">
Developed by Moushid
</p>

</center>

""",unsafe_allow_html=True)
