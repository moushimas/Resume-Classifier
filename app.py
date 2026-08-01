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
import numpy as np

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
# 3. LOAD MODEL AND VECTORIZER
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

    # Convert text to lowercase
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

    # Remove long numbers such as phone numbers
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

    # Split text
    words = text.split()

    # Remove stopwords and apply stemming
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

    st.subheader("📌 About Project")

    st.write(
        """
        This application analyzes resume content
        and predicts its professional category
        using Machine Learning and Natural
        Language Processing.
        """
    )

    st.divider()

    st.subheader("🤖 Model Details")

    st.write("**Algorithm:** Linear SVM")

    st.write("**Feature Extraction:** TF-IDF")

    st.write("**Categories:** 4 + Unknown")

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
    Upload a resume and the system will analyze
    its skills, technologies and professional
    information to predict the most suitable category.
    """
)

st.divider()


# ============================================================
# 9. SYSTEM OVERVIEW
# ============================================================

st.header(
    "📊 System Overview"
)

overview1, overview2, overview3 = st.columns(3)


with overview1:

    st.metric(
        label="Classification Model",
        value="Linear SVM"
    )


with overview2:

    st.metric(
        label="Trained Categories",
        value="4"
    )


with overview3:

    st.metric(
        label="Text Features",
        value="TF-IDF"
    )


st.divider()


# ============================================================
# 10. SUPPORTED CATEGORIES
# ============================================================

st.header(
    "🎯 Supported Resume Categories"
)

st.write(
    """
    The machine learning model is trained to identify
    the following resume categories.
    """
)


category1, category2, category3, category4 = st.columns(4)


with category1:

    st.subheader(
        "⚛️ React"
    )

    st.write(
        "**React Developer**"
    )

    st.caption(
        "Frontend and web application development"
    )


with category2:

    st.subheader(
        "🗄️ SQL"
    )

    st.write(
        "**SQL Developer**"
    )

    st.caption(
        "SQL and database development"
    )


with category3:

    st.subheader(
        "🏢 PeopleSoft"
    )

    st.write(
        "**PeopleSoft Resume**"
    )

    st.caption(
        "PeopleSoft and enterprise applications"
    )


with category4:

    st.subheader(
        "💼 Workday"
    )

    st.write(
        "**Workday**"
    )

    st.caption(
        "Workday and HR enterprise technologies"
    )


st.divider()


# ============================================================
# 11. UPLOAD RESUME
# ============================================================

st.header(
    "📤 Upload Resume"
)

st.write(
    "Upload your resume in PDF or DOCX format."
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

        # ====================================================
        # 13. EXTRACT RESUME TEXT
        # ====================================================

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


        # ====================================================
        # 14. CHECK EXTRACTED TEXT
        # ====================================================

        if not resume_text.strip():

            st.error(
                "No readable text was found in the resume."
            )

            st.warning(
                "Please upload a text-based PDF or DOCX file."
            )

            st.stop()


        # ====================================================
        # 15. RESUME INFORMATION
        # ====================================================

        st.divider()

        st.header(
            "📊 Resume Information"
        )


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

            total_words = len(
                resume_text.split()
            )

            st.metric(
                label="Total Words",
                value=total_words
            )


        with info3:

            total_characters = len(
                resume_text
            )

            st.metric(
                label="Characters",
                value=total_characters
            )


        with info4:

            file_size = (
                uploaded_file.size / 1024
            )

            st.metric(
                label="File Size",
                value=f"{file_size:.1f} KB"
            )


        # ====================================================
        # 16. RESUME PREVIEW
        # ====================================================

        st.subheader(
            "👁️ Resume Preview"
        )


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
        # 17. PREPROCESS RESUME
        # ====================================================

        with st.spinner(
            "Cleaning and preprocessing resume..."
        ):

            clean_resume = preprocess_text(
                resume_text
            )


        if not clean_resume.strip():

            st.error(
                "The resume does not contain enough useful text."
            )

            st.stop()


        # ====================================================
        # 18. TF-IDF TRANSFORMATION
        # ====================================================

        resume_vector = vectorizer.transform(
            [clean_resume]
        )


        # ====================================================
        # 19. LINEAR SVM PREDICTION
        # ====================================================

        with st.spinner(
            "🤖 AI is analyzing your resume..."
        ):

            scores = model.decision_function(
                resume_vector
            )


            # ----------------------------------------------
            # Handle multiclass Linear SVM
            # ----------------------------------------------

            if len(scores.shape) == 1:

                scores = scores.reshape(
                    1,
                    -1
                )


            # ----------------------------------------------
            # Find highest scoring class
            # ----------------------------------------------

            best_index = np.argmax(
                scores[0]
            )


            prediction = model.classes_[
                best_index
            ]


            best_score = float(
                scores[0][best_index]
            )


            # ----------------------------------------------
            # Unknown resume threshold
            # ----------------------------------------------

            threshold = 0.30


            if best_score < threshold:

                prediction = (
                    "Other / Unknown Resume"
                )


        # ====================================================
        # 20. CLASSIFICATION RESULT
        # ====================================================

        st.divider()

        st.header(
            "🎯 Classification Result"
        )


        result1, result2 = st.columns(
            [2, 1]
        )


        with result1:

            if prediction == "Other / Unknown Resume":

                st.warning(
                    "⚠️ This resume does not strongly match any trained category."
                )

                st.warning(
                    f"Predicted Category: {prediction}"
                )


            else:

                st.success(
                    f"✅ Predicted Category: {prediction}"
                )


            st.write(
                """
                The classification is based on the
                technical skills, technologies,
                experience and professional information
                detected in the uploaded resume.
                """
            )


        with result2:

            st.metric(
                label="Predicted Role",
                value=str(prediction)
            )

            st.metric(
                label="Decision Score",
                value=f"{best_score:.3f}"
            )


        # ====================================================
        # 21. DECISION SCORE INFORMATION
        # ====================================================

        if prediction == "Other / Unknown Resume":

            st.info(
                """
                The resume received a weak match with
                the four trained categories, so the
                system marked it as Other / Unknown.
                """
            )

        else:

            st.info(
                """
                The resume received a sufficient match
                with one of the categories learned by
                the Linear SVM model.
                """
            )


        # ====================================================
        # 22. SHOW CATEGORY SCORES
        # ====================================================

        st.subheader(
            "📈 Category Matching Scores"
        )


        score_data = {}


        for category, score in zip(
            model.classes_,
            scores[0]
        ):

            score_data[
                str(category)
            ] = float(score)


        st.bar_chart(
            score_data
        )


        with st.expander(
            "View individual category scores"
        ):

            for category, score in score_data.items():

                st.write(
                    f"{category}: {score:.3f}"
                )


        # ====================================================
        # 23. IMPORTANT INFORMATION
        # ====================================================

        st.warning(
            """
            Important: This model was trained mainly on
            React Developer, SQL Developer, PeopleSoft
            Resume and Workday resumes.

            A resume from a completely different profession
            may be marked as Other / Unknown Resume.
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
# 24. HOW THE SYSTEM WORKS
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
        Upload a resume
        in PDF or DOCX format.
        """
    )


with step2:

    st.subheader(
        "2️⃣ Preprocess"
    )

    st.write(
        """
        Resume text is extracted,
        cleaned and preprocessed.
        """
    )


with step3:

    st.subheader(
        "3️⃣ TF-IDF"
    )

    st.write(
        """
        TF-IDF converts resume
        text into numerical features.
        """
    )


with step4:

    st.subheader(
        "4️⃣ Classify"
    )

    st.write(
        """
        Linear SVM predicts the
        most suitable resume category.
        """
    )


# ============================================================
# 25. TECHNOLOGIES USED
# ============================================================

st.divider()

st.header(
    "🛠️ Technologies Used"
)


tech1, tech2, tech3, tech4 = st.columns(4)


with tech1:

    st.subheader(
        "🐍 Python"
    )

    st.write(
        "Application development"
    )


with tech2:

    st.subheader(
        "🧠 Scikit-learn"
    )

    st.write(
        "Machine Learning"
    )


with tech3:

    st.subheader(
        "🔤 NLP"
    )

    st.write(
        "Text preprocessing and TF-IDF"
    )


with tech4:

    st.subheader(
        "🌐 Streamlit"
    )

    st.write(
        "Web application deployment"
    )


# ============================================================
# 26. MODEL DETAILS
# ============================================================

st.divider()

st.header(
    "🤖 Model Details"
)


model1, model2, model3 = st.columns(3)


with model1:

    st.metric(
        label="Algorithm",
        value="Linear SVM"
    )


with model2:

    st.metric(
        label="Feature Extraction",
        value="TF-IDF"
    )


with model3:

    st.metric(
        label="Known Classes",
        value="4"
    )


# ============================================================
# 27. PROJECT INFORMATION
# ============================================================

st.divider()

st.header(
    "📌 Project Information"
)

st.write(
    """
    **Project:** AI Resume Classification System

    **Algorithm:** Linear Support Vector Machine

    **Feature Extraction:** TF-IDF Vectorization

    **NLP Processing:** Text cleaning, stopword removal and stemming

    **Web Application:** Streamlit
    """
)


# ============================================================
# 28. FOOTER
# ============================================================

st.divider()

st.subheader(
    "📄 AI Resume Classification System"
)

st.caption(
    "Machine Learning and Natural Language Processing Project"
)
