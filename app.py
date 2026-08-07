import streamlit as st
import pdfplumber
import docx2txt
import joblib
import nltk
import re
import string
import numpy as np

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

/* ================= MAIN APP ================= */

.stApp{
    background:#F4F8FC;
}

/* Main container */

.block-container{
    max-width:1200px;
    padding-top:2rem;
    padding-bottom:2rem;
}

/* ================= TEXT ================= */

html,
body,
p,
label,
span,
div{
    color:#1F2937;
}

h1,h2,h3,h4,h5{
    color:#0F172A;
    font-weight:700;
}

/* ================= SIDEBAR ================= */

section[data-testid="stSidebar"]{
    background:#1E3A8A;
}

section[data-testid="stSidebar"] *{
    color:white !important;
}

/* ================= METRICS ================= */

div[data-testid="stMetric"]{

    background:#FFFFFF;

    border-radius:18px;

    padding:18px;

    border:1px solid #E5E7EB;

    box-shadow:0px 5px 15px rgba(0,0,0,.08);

}

/* ================= BUTTON ================= */

.stButton>button{

    background:#2563EB;

    color:white;

    border:none;

    border-radius:12px;

    height:48px;

    font-size:17px;

}

.stButton>button:hover{

    background:#1D4ED8;

}

/* ================= FILE UPLOADER ================= */

[data-testid="stFileUploader"]{

    background:#FFFFFF;

    border-radius:18px;

    border:2px dashed #2563EB;

    padding:18px;

}

/* ================= TEXT AREA ================= */

textarea{

    color:#111827 !important;

    background:#FFFFFF !important;

}

/* ================= EXPANDER ================= */

details{

    background:#FFFFFF;

    border-radius:12px;

    padding:10px;

}

/* ================= ALERTS ================= */

div[data-baseweb="notification"]{

    color:#111827 !important;

}

/* ================= INPUT ================= */

input{

    color:#111827 !important;

}

/* ================= MARKDOWN ================= */

[data-testid="stMarkdownContainer"]{

    color:#111827;

}

</style>
""", unsafe_allow_html=True)

 # -------------------------------------------------------
# NLTK
# -------------------------------------------------------

try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    stop_words = set(stopwords.words("english"))

stemmer = PorterStemmer()
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

# -------------------------------------------------
# TEXT PREPROCESSING (Moved from another cell, if not already present here)
# -------------------------------------------------

def preprocess_text(text):

    text=str(text).lower()

    text=re.sub(r"http\S+|www\S+"," ",text)

    text=re.sub(r"\S+@\S+"," ",text)

    text=re.sub(r"\d{10,}"," ",text)

    text=text.translate(
        str.maketrans(
            string.punctuation,
            " "*len(string.punctuation)
        )
    )

    words=text.split()

    words=[
        stemmer.stem(word)
        for word in words
        if word not in stop_words
        and len(word)>1
    ]

    return " ".join(words)

# -------------------------------------------------
# PDF EXTRACTION (Moved from another cell, if not already present here)
# -------------------------------------------------

def extract_pdf(file):

    text=""

    with pdfplumber.open(file) as pdf:

        for page in pdf.pages:

            page_text=page.extract_text()

            if page_text:

                text+=page_text+"\n"

    return text

# -------------------------------------------------
# DOCX EXTRACTION (Moved from another cell, if not already present here)
# -------------------------------------------------

def extract_docx(file):

    return docx2txt.process(file)


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

# =======================================================
# PREMIUM HERO HEADER
# =======================================================

st.markdown("""

<div style="
background:linear-gradient(135deg,#2563EB,#1D4ED8,#0EA5E9);
padding:40px;
border-radius:25px;
box-shadow:0px 10px 25px rgba(0,0,0,.20);
">

<h1 style="
text-align:center;
color:white;
font-size:52px;
font-weight:bold;
margin-bottom:10px;
">

🤖 AI Resume Classification System

</h1>

<p style="
text-align:center;
font-size:22px;
color:white;
">

Smart Resume Analysis using Machine Learning,
Natural Language Processing and Linear SVM

</p>

</div>

""",unsafe_allow_html=True)

st.write("")

# =======================================================
# BEAUTIFUL DASHBOARD CARDS
# =======================================================

c1,c2,c3=st.columns(3)

with c1:

    st.markdown("""

<div style="
background:white;
padding:25px;
border-radius:20px;
text-align:center;
box-shadow:0px 6px 15px rgba(0,0,0,.08);
">

<h1>🤖</h1>

<h3 style="color:#2563EB;">
Linear SVM
</h3>

<p>
Machine Learning Model
</p>

</div>

""",unsafe_allow_html=True)

with c2:

    st.markdown("""

<div style="
background:white;
padding:25px;
border-radius:20px;
text-align:center;
box-shadow:0px 6px 15px rgba(0,0,0,.08);
">

<h1>📄</h1>

<h3 style="color:#2563EB;">
4 Categories
</h3>

<p>
Resume Classification
</p>

</div>

""",unsafe_allow_html=True)

with c3:

    st.markdown("""

<div style="
background:white;
padding:25px;
border-radius:20px;
text-align:center;
box-shadow:0px 6px 15px rgba(0,0,0,.08);
">

<h1>🧠</h1>

<h3 style="color:#2563EB;">
TF-IDF
</h3>

<p>
Feature Extraction
</p>

</div>

""",unsafe_allow_html=True)

st.markdown("<br>",unsafe_allow_html=True)
    # ============================================================
# 11. UPLOAD RESUME
# ============================================================

# ============================================================
# PREMIUM UPLOAD SECTION
# ============================================================

st.markdown("""

<div style="
background:white;
padding:35px;
border-radius:25px;
box-shadow:0px 10px 25px rgba(0,0,0,.08);
">

<h2 style="
text-align:center;
color:#2563EB;
">

📤 Upload Resume

</h2>

<p style="
text-align:center;
font-size:18px;
color:#475569;
">

Upload your PDF or DOCX resume and let our AI
predict the most suitable job category.

</p>

</div>

""", unsafe_allow_html=True)

st.write("")

uploaded_file = st.file_uploader(
    "📂 Drag & Drop Resume Here",
    type=["pdf","docx"]
)

st.write("")

if uploaded_file:

    st.balloons()

    st.success(f"✅ {uploaded_file.name} Uploaded Successfully")

    st.markdown("""
<div style="
background:#ECFDF5;
padding:18px;
border-radius:15px;
border-left:8px solid #10B981;
">

<h3 style="color:#047857;">

✔ Resume Ready For Analysis

</h3>

</div>

""",unsafe_allow_html=True)

    # --- ADDED: Logic to extract resume text based on file type ---
    file_extension = uploaded_file.name.split(".")[-1].lower()
    resume_text = ""
    if file_extension == "pdf":
        resume_text = extract_pdf(uploaded_file)
    elif file_extension == "docx":
        resume_text = extract_docx(uploaded_file)
    else:
        st.error("Unsupported file type. Please upload a PDF or DOCX file.")
        st.stop()

    # --------------------------------------------
    # Validate Resume
    # --------------------------------------------

    if not resume_text.strip():

        st.error("Resume text not found.")

        st.stop()

    # --- ADDED: Preprocess resume text ---
    clean_resume = preprocess_text(resume_text)

    st.markdown("---")

    st.markdown("""
    <h2 style="color:#1E40AF;">
    📊 Resume Statistics
    </h2>
    """,unsafe_allow_html=True)

    file_type = uploaded_file.name.split(".")[-1].upper()

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("📄 File Type", file_type)
    c2.metric("📝 Words", len(resume_text.split()))
    c3.metric("🔤 Characters", len(resume_text))
    c4.metric("💾 Size", f"{uploaded_file.size/1024:.1f} KB")

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
    # --- ADDED: Make prediction ---
    scores = model.decision_function(resume_vector)
    best_index = np.argmax(scores[0])
    prediction = model.classes_[best_index]

    st.markdown("""

<h2 style="
text-align:center;
color:#1E3A8A;
">

🎯 AI Prediction

</h2>

""",unsafe_allow_html=True)

    st.markdown(f"""

<div style="
background:linear-gradient(135deg,#10B981,#059669);
padding:40px;
border-radius:25px;
box-shadow:0px 10px 25px rgba(0,0,0,.20);
text-align:center;
">

<h3 style="color:white;">

Predicted Category

</h3>

<h1 style="
font-size:48px;
color:white;
">

{prediction}

</h1>

<p style="
font-size:20px;
color:white;
">

Classification Completed Successfully

</p>

</div>

""",unsafe_allow_html=True)

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
