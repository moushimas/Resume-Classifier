
# ============================================================
# AI RESUME CLASSIFICATION SYSTEM
# ============================================================

import streamlit as st
import pdfplumber
import docx2txt
import joblib
import nltk
import re
import string

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Resume Classifier",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. NLTK SETUP
# ============================================================

try:
    stop_words = set(stopwords.words("english"))

except LookupError:
    nltk.download("stopwords", quiet=True)
    stop_words = set(stopwords.words("english"))

stemmer = PorterStemmer()


# ============================================================
# 3. LOAD MODEL AND TF-IDF VECTORIZER
# ============================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        "resume_classifier_model.pkl"
    )

    vectorizer = joblib.load(
        "tfidf_vectorizer.pkl"
    )

    return model, vectorizer


try:
    model, vectorizer = load_model()

except Exception as e:

    st.error(
        f"Model loading error: {e}"
    )

    st.stop()


# ============================================================
# 4. TEXT PREPROCESSING
# ============================================================

def preprocess_text(text):

    # Convert to lowercase
    text = str(text).lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    # Remove email addresses
    text = re.sub(
        r"\S+@\S+",
        " ",
        text
    )

    # Remove long numbers
    text = re.sub(
        r"\d{10,}",
        " ",
        text
    )

    # Remove punctuation
    text = text.translate(
        str.maketrans(
            string.punctuation,
            " " * len(string.punctuation)
        )
    )

    # Split text into words
    words = text.split()

    # Remove stopwords and perform stemming
    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
        and len(word) > 1
    ]

    return " ".join(words)


# ============================================================
# 5. PDF TEXT EXTRACTION
# ============================================================

def extract_pdf(file):

    text = ""

    with pdfplumber.open(file) as pdf:

        for page in pdf.pages:

            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

    return text


# ============================================================
# 6. DOCX TEXT EXTRACTION
# ============================================================

def extract_docx(file):

    return docx2txt.process(file)


# ============================================================
# 7. SIDEBAR
# ============================================================

with st.sidebar:

    st.title("📄 Resume AI")

    st.caption(
        "Machine Learning Resume Classification"
    )

    st.divider()

    st.subheader("📌 About")

    st.write(
        """
        This application automatically analyzes
        resumes and predicts their professional
        category using Machine Learning and
        Natural Language Processing.
        """
    )

    st.divider()

    st.subheader("🤖 Model Details")

    st.write("**Algorithm:** Linear SVM")

    st.write("**Feature Extraction:** TF-IDF")

    st.write("**Number of Categories:** 4")

    st.divider()

    st.subheader("📁 Supported Files")

    st.write("📕 PDF")

    st.write("📘 DOCX")

    st.divider()

    st.info(
        "Upload a resume to start classification."
    )


# ============================================================
# 8. MAIN HEADER
# ============================================================

st.title(
    "📄 AI Resume Classification System"
)

st.subheader(
    "Smart Resume Analysis Using Machine Learning"
)

st.write(
    """
    Upload a resume and the system will automatically
    analyze its skills, technologies and professional
    content to predict the most suitable resume category.
    """
)

st.divider()


# ============================================================
# 9. MODEL INFORMATION
# ============================================================

st.header("📊 System Overview")

overview1, overview2, overview3 = st.columns(3)

with overview1:

    st.metric(
        label="Classification Model",
        value="Linear SVM"
    )

with overview2:

    st.metric(
        label="Resume Categories",
        value="4"
    )

with overview3:

    st.metric(
        label="Text Technique",
        value="TF-IDF"
    )


st.divider()


# ============================================================
# 10. SUPPORTED CATEGORIES
# ============================================================

st.header("🎯 Supported Resume Categories")

st.write(
    "The model is trained to classify resumes into the following four categories."
)

category1, category2, category3, category4 = st.columns(4)


with category1:

    st.subheader("⚛️ React")

    st.write("React Developer")

    st.caption(
        "Frontend and web application development"
    )


with category2:

    st.subheader("🗄️ SQL")

    st.write("SQL Developer")

    st.caption(
        "Database development and SQL technologies"
    )


with category3:

    st.subheader("🏢 PeopleSoft")

    st.write("PeopleSoft Resume")

    st.caption(
        "PeopleSoft and enterprise applications"
    )


with category4:

    st.subheader("💼 Workday")

    st.write("Workday")

    st.caption(
        "Workday and enterprise HR technologies"
    )


st.divider()


# ============================================================
# 11. RESUME UPLOAD SECTION
# ============================================================

st.header("📤 Upload Resume")

st.write(
    """
    Select a resume from your computer.

    Supported formats: **PDF and DOCX**
    """
)


uploaded_file = st.file_uploader(
    "Choose your resume",
    type=["pdf", "docx"]
)


# ============================================================
# 12. PROCESS UPLOADED RESUME
# ============================================================

if uploaded_file is not None:

    st.success(
        f"✅ Resume uploaded successfully: {uploaded_file.name}"
    )

    try:

        # ----------------------------------------------------
        # Extract resume text
        # ----------------------------------------------------

        if uploaded_file.name.lower().endswith(".pdf"):

            with st.spinner(
                "Reading PDF resume..."
            ):

                resume_text = extract_pdf(
                    uploaded_file
                )


        elif uploaded_file.name.lower().endswith(".docx"):

            with st.spinner(
                "Reading DOCX resume..."
            ):

                resume_text = extract_docx(
                    uploaded_file
                )


        else:

            st.error(
                "Unsupported file format."
            )

            st.stop()


        # ----------------------------------------------------
        # Check extracted text
        # ----------------------------------------------------

        if not resume_text.strip():

            st.error(
                "No readable text was found in the uploaded resume."
            )

            st.warning(
                "Please upload a text-based PDF or DOCX resume."
            )

            st.stop()


        # ====================================================
        # 13. RESUME INFORMATION
        # ====================================================

        st.divider()

        st.header("📊 Resume Information")


        info1, info2, info3, info4 = st.columns(4)


        with info1:

            file_type = (
                uploaded_file.name
                .split(".")[-1]
                .upper()
            )

            st.metric(
                label="File Type",
                value=file_type
            )


        with info2:

            st.metric(
                label="Total Words",
                value=len(
                    resume_text.split()
                )
            )


        with info3:

            st.metric(
                label="Characters",
                value=len(
                    resume_text
                )
            )


        with info4:

            st.metric(
                label="File Size",
                value=f"{uploaded_file.size / 1024:.1f} KB"
            )


        # ====================================================
        # 14. RESUME PREVIEW
        # ====================================================

        st.subheader("👁️ Resume Preview")


        with st.expander(
            "Click here to view extracted resume text"
        ):

            st.text_area(
                "Extracted Resume Content",
                value=resume_text[:5000],
                height=300,
                disabled=True
            )


        # ====================================================
        # 15. PREPROCESS RESUME
        # ====================================================

        with st.spinner(
            "Cleaning and preprocessing resume..."
        ):

            clean_resume = preprocess_text(
                resume_text
            )


        if not clean_resume.strip():

            st.error(
                "The resume does not contain enough useful text for classification."
            )

            st.stop()


        # ====================================================
        # 16. TF-IDF TRANSFORMATION
        # ====================================================

        resume_vector = vectorizer.transform(
            [clean_resume]
        )


        # ====================================================
        # 17. PREDICTION

        with st.spinner(
            "🤖 AI is analyzing your resume..."
        ):

            prediction = model.predict(
                resume_vector
            )[0]


        # ====================================================
        # 18. CLASSIFICATION RESULT
        # ====================================================

        st.divider()

        st.header(
            "🎯 Classification Result"
        )


        result1, result2 = st.columns(
            [2, 1]
        )


        with result1:

            st.success(
                f"Predicted Category: {prediction}"
            )

            st.write(
                """
                The prediction was generated by analyzing
                the technical skills, experience,
                technologies and professional information
                contained in the uploaded resume.
                """
            )


        with result2:

            st.metric(
                label="Predicted Role",
                value=str(prediction)
            )


        # ====================================================
        # 19. MODEL INFORMATION
        # ====================================================

        st.info(
            """
            ℹ️ The classifier can only predict one of the
            four categories on which it was trained:
            React Developer, SQL Developer,
            PeopleSoft Resume or Workday.
            """
        )


    except Exception as e:

        st.error(
            "An error occurred while processing the resume."
        )

        st.write(
            f"Error details: {e}"
        )


# ============================================================
# 20. HOW THE SYSTEM WORKS
# ============================================================

st.divider()

st.header(
    "⚙️ How The System Works"
)


step1, step2, step3, step4 = st.columns(4)


with step1:

    st.subheader(
        "1️⃣ Upload"
    )

    st.write(
        """
        Upload a resume in
        PDF or DOCX format.
        """
    )


with step2:

    st.subheader(
        "2️⃣ Preprocess"
    )

    st.write(
        """
        The resume text is
        extracted and cleaned.
        """
    )


with step3:

    st.subheader(
        "3️⃣ TF-IDF"
    )

    st.write(
        """
        TF-IDF converts the
        resume text into
        numerical features.
        """
    )


with step4:

    st.subheader(
        "4️⃣ Predict"
    )

    st.write(
        """
        Linear SVM analyzes
        the features and predicts
        the resume category.
        """
    )


# ============================================================
# 21. TECHNOLOGIES USED
# ============================================================

st.divider()

st.header(
    "🛠️ Technologies Used"
)


tech1, tech2, tech3 = st.columns(3)


with tech1:

    st.subheader(
        "🐍 Python"
    )

    st.write(
        "Programming and application development"
    )


with tech2:

    st.subheader(
        "🧠 Machine Learning"
    )

    st.write(
        "Scikit-learn, TF-IDF and Linear SVM"
    )


with tech3:

    st.subheader(
        "🌐 Streamlit"
    )

    st.write(
        "Interactive web application and deployment"
    )


# ============================================================
# 22. PROJECT INFORMATION
# ============================================================

st.divider()

st.header(
    "📌 Project Information"
)

st.write(
    """
    **Project:** AI Resume Classification System

    **Machine Learning Algorithm:** Linear Support Vector Machine

    **Feature Extraction:** TF-IDF Vectorization

    **Natural Language Processing:** Text cleaning, stopword removal and stemming

    **Web Framework:** Streamlit
    """
)


# ============================================================
# 23. FOOTER
# ============================================================

st.divider()

st.subheader(
    "📄 AI Resume Classification System"
)

st.caption(
    "Machine Learning and Natural Language Processing Project"
)
