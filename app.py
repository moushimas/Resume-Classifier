import streamlit as st
import docx2txt
import joblib
import nltk
import re
import string

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------

st.set_page_config(
    page_title="AI Resume Classification",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------
# CUSTOM CSS
# -------------------------------------------------

st.markdown("""
<style>

.stApp{
background:linear-gradient(135deg,#eef4ff,#f7fbff,#ffffff);
}

.block-container{
padding-top:2rem;
padding-bottom:2rem;
}

h1,h2,h3{
color:#0F172A;
}

[data-testid="stMetric"]{
background:white;
padding:15px;
border-radius:15px;
box-shadow:0px 5px 15px rgba(0,0,0,.08);
}

.stButton>button{
background:#2563EB;
color:white;
border:none;
border-radius:10px;
height:48px;
width:100%;
font-size:18px;
font-weight:bold;
}

.stButton>button:hover{
background:#1D4ED8;
}

.css-1d391kg{
background:#0F172A;
}

</style>
""",unsafe_allow_html=True)

# -------------------------------------------------
# NLTK
# -------------------------------------------------

try:
    stop_words=set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords",quiet=True)
    stop_words=set(stopwords.words("english"))

stemmer=PorterStemmer()

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------

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
# TEXT PREPROCESSING
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
# PDF
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
# DOCX
# -------------------------------------------------

def extract_docx(file):

    return docx2txt.process(file)

# -------------------------------------------------
# SIDEBAR
# -------------------------------------------------

with st.sidebar:

    st.image(
        "https://img.icons8.com/color/96/artificial-intelligence.png",
        width=90
    )

    st.title("AI Resume")

    st.success("Machine Learning Project")

    st.markdown("---")

    st.markdown("### 📌 About")

    st.write("""
This application classifies resumes using
Machine Learning and NLP.
""")

    st.markdown("---")

    st.markdown("### 🤖 Model")

    st.info("""
Algorithm : Linear SVM

Feature : TF-IDF

Categories : 4
""")

    st.markdown("---")

    st.markdown("### 📁 File Types")

    st.write("✅ PDF")

    st.write("✅ DOCX")

# -------------------------------------------------
# HEADER
# -------------------------------------------------

st.markdown("""
<h1 style='text-align:center;
color:#1E3A8A;
font-size:46px;
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
    st.metric("📂 Categories","4")

with col3:
    st.metric("🧠 Technique","TF-IDF")
    
