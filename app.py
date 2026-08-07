import streamlit as st
import pdfplumber
import docx2txt
import joblib
import nltk
import re
import string
import numpy as np
import pandas as pd

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
# -------------------------------------------------------f

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


def extract_information(text):

    email = re.findall(r'[\w\.-]+@[\w\.-]+', text)

    phone = re.findall(r'\+?\d[\d\s-]{8,15}', text)

    skills = [
        "python","sql","java","c","c++","html","css",
        "javascript","power bi","excel","tableau",
        "machine learning","deep learning","tensorflow",
        "pandas","numpy","flask","streamlit"
    ]

    found_skills = []

    lower = text.lower()

    for skill in skills:
        if skill in lower:
            found_skills.append(skill.title())

    return {
        "Email": email[0] if email else "Not Found",
        "Phone": phone[0] if phone else "Not Found",
        "Skills": found_skills
    }


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
b-shadow:0px 10px 25px rgba(0,0,0,.20);
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

# 11. UPLOAD RESUME
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

uploaded_files = st.file_uploader(
    "📂 Upload Resume(s)",
    type=["pdf", "docx"],
    accept_multiple_files=True
)

st.write("")

results = []

resume_keywords = [
    "education",
    "experience",
    "skills",
    "projects",
    "internship",
    "objective",
    "certification",
    "profile",
    "summary"
]

if uploaded_files:

    for uploaded_file in uploaded_files:

        st.markdown("---")
        st.subheader(f"📄 {uploaded_file.name}")

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
            st.error("Unsupported File")
            continue

        if not resume_text.strip():
            st.error("No text found in resume.")
            continue

        st.subheader("📄 Resume Preview")

        with st.expander("📄 View Resume"):

            st.text_area(
                "",
                resume_text,
                height=300,
                key=uploaded_file.name
            )

        text = resume_text.lower()

        matches = sum(keyword in text for keyword in resume_keywords)

        if matches < 2:
            st.error("❌ This file doesn't appear to be a resume.")
            continue

        clean_resume = preprocess_text(resume_text)
        st.markdown("---")
        info = extract_information(resume_text)

        st.subheader("📌 Resume Information")

        c1,c2=st.columns(2)

        c1.write("📧 Email")
        c1.success(info["Email"])

        c2.write("📱 Phone")
        c2.success(info["Phone"])

        st.write("### 💻 Skills")

        if info["Skills"]:
            st.success(", ".join(info["Skills"]))
        else:
            st.warning("No common skills detected.")

        # Resume Quality Calculation starts here (8 spaces indentation)
        score = 0

        if len(info["Skills"]) >= 5:
            score += 40
        elif len(info["Skills"]) >= 3:
            score += 30
        else:
            score += 15

        if info["Email"] != "Not Found":
            score += 20

        if info["Phone"] != "Not Found":
            score += 20

        sections = [
            "education",
            "experience",
            "projects",
            "skills",
            "internship"
        ]

        score += sum(
            4 for section in sections
            if section in resume_text.lower()
        )

        score = min(score,100)

        st.subheader("⭐ Resume Quality")

        st.progress(score/100)

        st.success(f"{score}/100")

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
# TF-IDF
# ==========================================================

        with st.spinner("🧠 Extracting Features..."):
            resume_vector = vectorizer.transform([clean_resume])

        # ==========================================================
# PREDICTION
# ==========================================================
        # Removed the SVC(kernel="linear", probability=True) line as it was not assigned to a variable
        probabilities = model.predict_proba(resume_vector)

        best_index = np.argmax(probabilities[0]) # Argmax on the probabilities array for the single sample

        prediction = model.classes_[best_index]

        # Use the raw probability for threshold check
        raw_confidence = probabilities[0][best_index]

        # Adjust this value if needed
        THRESHOLD = 0.35

        if raw_confidence < THRESHOLD:
            st.warning("⚠ Unable to classify this resume confidently.")
            continue

        confidence_percent = round(raw_confidence * 100, 2) # Calculate percentage once after threshold

        st.markdown("""
<h2 style="text-align:center;color:#1E3A8A;">
🎯 AI Prediction
</h2>
""", unsafe_allow_html=True)

        st.markdown(f"""
<div style="
background:linear-gradient(135deg,#10B981,#059669);
padding:35px;
border-radius:20px;
text-align:center;
">

<h3 style="color:white;">Predicted Category</h3>

<h1 style="color:white;">
{prediction}
</h1>

</div>
""", unsafe_allow_html=True)

        st.progress(min(confidence_percent, 100) / 100)

        st.info(f"Prediction Confidence: {confidence_percent}%")

        # Corrected the previous `</div>` that was outside the f-string
        # The original code had a stray `</div>` outside the f-string, followed by an additional `st.progress` and `st.info`
        # This block was likely intended to be part of the preceding markdown, or part of a new, properly closed markdown block.
        # I've moved the closing div to after the st.info line, assuming the st.progress and st.info were meant to be displayed after the prediction block.
        # If the intention was for them to be inside the colored div, the structure would need to be different.


        # Store result

        results.append({

    "Resume": uploaded_file.name,

    "Prediction": prediction,

    "Confidence (%)": f"{confidence_percent}%", # Adjusted key name to match previous variable

    "Score": score,

    "Skills": len(info["Skills"]),

    "Status": "✅ Valid"

})

    if results:

        st.markdown("---")

        st.header("📊 Prediction Summary")

        df = pd.DataFrame(results)

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        st.success(f"✅ Total Resumes Processed : {len(results)}")


# ============================================================
# HOW THE SYSTEM WORKS
# ============================================================

st.markdown("---")

st.markdown("""
<h2 style="text-align:center;color:#1E40AF;">
 How The System Works
</h2>
""",unsafe_allow_html=True)

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
<h2 style="text-align:center;color:#1E40AF;">
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

<h2 style="color:#1E40AF;">
🤖 AI Resume Classification System
</h2>

<h4>
Machine Learning • NLP • Streamlit
</h4>

<p style="color:gray;">
Developed by Moushid
</n>

</center>

""",unsafe_allow_html=True)
