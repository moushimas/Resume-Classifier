!pip install -q streamlit
import streamlit as st
import pdfplumber
import docx2txt
import joblib
import nltk
import re
import string

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


st.set_page_config(
    page_title="Resume Classification System",
    page_icon="📄",
    layout="wide"
)


# -------------------------
# NLTK
# -------------------------

try:
    stop_words = set(
        stopwords.words("english")
    )
except LookupError:
    nltk.download("stopwords")
    stop_words = set(
        stopwords.words("english")
    )

stemmer = PorterStemmer()


# -------------------------
# Load Model
# -------------------------

@st.cache_resource
def load_model():

    model = joblib.load(
        "resume_classifier_model.pkl"
    )

    vectorizer = joblib.load(
        "tfidf_vectorizer.pkl"
    )

    return model, vectorizer


model, vectorizer = load_model()


# -------------------------
# Preprocessing
# MUST MATCH TRAINING
# -------------------------

def preprocess_text(text):

    text = str(text).lower()

    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    text = re.sub(
        r"\S+@\S+",
        " ",
        text
    )

    text = re.sub(
        r"\d{10,}",
        " ",
        text
    )

    text = text.translate(
        str.maketrans(
            string.punctuation,
            " " * len(string.punctuation)
        )
    )

    words = text.split()

    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
        and len(word) > 1
    ]

    return " ".join(words)


# -------------------------
# PDF
# -------------------------

def extract_pdf(file):

    text = ""

    with pdfplumber.open(file) as pdf:

        for page in pdf.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    return text


# -------------------------
# DOCX
# -------------------------

def extract_docx(file):

    return docx2txt.process(file)


# -------------------------
# UI
# -------------------------

st.title(
    "📄 Resume Classification System"
)

st.write(
    """
Upload a resume and classify it into one of
the following categories:

• React Developer  
• Peoplesoft Resume  
• Workday  
• SQL Developer
"""
)


uploaded_file = st.file_uploader(
    "Upload Resume",
    type=["pdf", "docx"]
)


if uploaded_file is not None:

    try:

        if uploaded_file.name.lower().endswith(
            ".pdf"
        ):

            resume = extract_pdf(
                uploaded_file
            )

        else:

            resume = extract_docx(
                uploaded_file
            )


        if not resume.strip():

            st.error(
                "Could not extract text from the resume."
            )

            st.stop()


        st.subheader(
            "Resume Preview"
        )

        st.text_area(
            "Extracted Resume Text",
            resume[:3000],
            height=250
        )


        clean_resume = preprocess_text(
            resume
        )

        vector = vectorizer.transform(
            [clean_resume]
        )


        prediction = model.predict(
            vector
        )[0]


        st.success(
            f"Predicted Category: {prediction}"
        )


    except Exception as e:

        st.error(
            f"Error: {e}"
        )
