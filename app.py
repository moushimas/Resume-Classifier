
import streamlit as st
import pdfplumber
import docx2txt
import joblib
import nltk
import re
import string

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# -----------------------------
# Download NLTK data (only if missing)
# -----------------------------
import nltk

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# -----------------------------
# Streamlit Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Resume Classification System",
    page_icon="📄",
    layout="wide"
)

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
 try:
    model = joblib.load("svm_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
except Exception as e:
    st.error(f"Model loading error: {e}")
    st.stop()
    return model, vectorizer

try:
    model, vectorizer = load_model()
except Exception as e:
    st.error(f"Unable to load model: {e}")
    st.stop()

# -----------------------------
# Preprocessing
# -----------------------------
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def preprocess_text(text):

    text = text.lower()

    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"\d+", " ", text)

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    words = text.split()

    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# -----------------------------
# Read PDF
# -----------------------------
def extract_pdf(file):

    text = ""

    with pdfplumber.open(file) as pdf:

        for page in pdf.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    return text

# -----------------------------
# Read DOCX
# -----------------------------
def extract_docx(file):

    return docx2txt.process(file)

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("Resume Classifier")

st.sidebar.info(
    """
Upload a resume in PDF or DOCX format.

The model predicts the resume category using
a trained SVM classifier.
"""
)

# -----------------------------
# Main UI
# -----------------------------
st.title("📄 Resume Classification System")

st.write("Upload a Resume (PDF or DOCX)")

uploaded_file = st.file_uploader(
    "Choose Resume",
    type=["pdf", "docx"]
)

if uploaded_file is not None:

    if uploaded_file.name.endswith(".pdf"):
        resume = extract_pdf(uploaded_file)
    else:
        resume = extract_docx(uploaded_file)

    st.subheader("Resume Preview")

    st.text_area(
        "Resume Content",
        resume[:3000],
        height=250
    )

    clean_resume = preprocess_text(resume)

    with st.spinner("Predicting..."):

        try:

            vector = vectorizer.transform([clean_resume])

            prediction = model.predict(vector)[0]

            st.success(
                f"Predicted Category: {prediction}"
            )

        except Exception as e:

            st.error(
                f"Prediction Error: {e}"
            )
