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
# 3. LOAD MODEL
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
    st.error(f"Model loading error: {e}")
    st.stop()


# ============================================================
# 4. PREPROCESS TEXT
# ============================================================

def preprocess_text(text):

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

    words = text.split()

    words = [
        stemmer.stem(word)
        for word in words
        if word not in stop_words
        and len(word) > 1
    ]

    return " ".join(words)


# ============================================================
# 5. EXTRACT PDF
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
# 6. EXTRACT DOCX
# ============================================================

def extract_docx(file):

    return docx2txt.process(file)


# ============================================================
# 7. PREDICTION FUNCTION
# ============================================================

def predict_resume(clean_resume):

    # Convert text into TF-IDF features
    resume_vector = vectorizer.transform(
        [clean_resume]
    )

    # Get SVM decision scores
    scores = model.decision_function(
        resume_vector
    )

    scores = np.asarray(scores)

    if scores.ndim == 1:
        scores = scores.reshape(1, -1)

    # Find highest scoring category
    best_index = int(
        np.argmax(scores[0])
    )

    predicted_category = model.classes_[
        best_index
    ]

    best_score = float(
        scores[0][best_index]
    )

    # Sort scores from highest to lowest
    sorted_scores = np.sort(
        scores[0]
    )[::-1]

    # Difference between best and second-best class
    if len(sorted_scores) > 1:
        score_margin = (
            sorted_scores[0] -
            sorted_scores[1]
        )
    else:
        score_margin = 0.0

    # --------------------------------------------------------
    # UNKNOWN DETECTION
    # --------------------------------------------------------

    # Starting values only.
    # These should later be tuned using validation resumes.

    minimum_score = 0.30
    minimum_margin = 0.10

    if (
        best_score < minimum_score
        or score_margin < minimum_margin
    ):

        final_prediction = (
            "Other / Unknown Resume"
        )

    else:

        final_prediction = (
            predicted_category
        )

    return (
        final_prediction,
        predicted_category,
        best_score,
        score_margin,
        scores[0]
    )


# ============================================================
# 8. SIDEBAR
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
        This application analyzes resumes
        using Natural Language Processing
        and Machine Learning.
        """
    )

    st.divider()

    st.subheader("🤖 Model")

    st.write(
        "**Algorithm:** Linear SVM"
    )

    st.write(
        "**Features:** TF-IDF"
    )

    st.write(
        "**Known Categories:** 4"
    )

    st.divider()

    st.subheader("📁 File Formats")

    st.write("📕 PDF")

    st.write("📘 DOCX")

    st.divider()

    st.info(
        """
        Resumes outside the trained categories
        may be classified as Other / Unknown.
        """
    )


# ============================================================
# 9. HEADER
# ============================================================

st.title(
    "📄 AI Resume Classification System"
)

st.subheader(
    "Resume Classification Using NLP and Machine Learning"
)

st.write(
    """
    Upload a resume and the system will analyze
    its skills, technologies and professional
    content to determine whether it matches
    one of the trained resume categories.
    """
)

st.divider()


# ============================================================
# 10. SYSTEM OVERVIEW
# ============================================================

st.header(
    "📊 System Overview"
)

col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "Machine Learning Model",
        "Linear SVM"
    )


with col2:

    st.metric(
        "Known Categories",
        "4"
    )


with col3:

    st.metric(
        "Feature Extraction",
        "TF-IDF"
    )


st.divider()


# ============================================================
# 11. TRAINED CATEGORIES
# ============================================================

st.header(
    "🎯 Trained Resume Categories"
)


category1, category2, category3, category4 = st.columns(4)


with category1:

    st.subheader("⚛️ React")

    st.write(
        "**React Developer**"
    )

    st.caption(
        "Frontend and web development"
    )


with category2:

    st.subheader("🗄️ SQL")

    st.write(
        "**SQL Developer**"
    )

    st.caption(
        "Database and SQL development"
    )


with category3:

    st.subheader("🏢 PeopleSoft")

    st.write(
        "**PeopleSoft Resume**"
    )

    st.caption(
        "PeopleSoft enterprise applications"
    )


with category4:

    st.subheader("💼 Workday")

    st.write(
        "**Workday**"
    )

    st.caption(
        "Workday enterprise technologies"
    )


st.info(
    """
    Resumes that do not strongly match these categories
    can be returned as Other / Unknown Resume.
    """
)

st.divider()


# ============================================================
# 12. UPLOAD
# ============================================================

st.header(
    "📤 Upload Resume"
)

st.write(
    "Upload a PDF or DOCX resume."
)


uploaded_file = st.file_uploader(
    "Choose Resume",
    type=["pdf", "docx"]
)


# ============================================================
# 13. PROCESS RESUME
# ============================================================

if uploaded_file is not None:

    st.success(
        f"Uploaded successfully: {uploaded_file.name}"
    )

    try:

        # ----------------------------------------------------
        # Extract text
        # ----------------------------------------------------

        if uploaded_file.name.lower().endswith(
            ".pdf"
        ):

            with st.spinner(
                "Reading PDF..."
            ):

                resume_text = extract_pdf(
                    uploaded_file
                )

        else:

            with st.spinner(
                "Reading DOCX..."
            ):

                resume_text = extract_docx(
                    uploaded_file
                )


        # ----------------------------------------------------
        # Validate extracted text
        # ----------------------------------------------------

        if not resume_text.strip():

            st.error(
                "No readable text found in the resume."
            )

            st.stop()


        # ====================================================
        # 14. RESUME INFORMATION
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
                "File Type",
                file_type
            )


        with info2:

            st.metric(
                "Words",
                len(resume_text.split())
            )


        with info3:

            st.metric(
                "Characters",
                len(resume_text)
            )


        with info4:

            st.metric(
                "File Size",
                f"{uploaded_file.size / 1024:.1f} KB"
            )


        # ====================================================
        # 15. RESUME PREVIEW
        # ====================================================

        st.subheader(
            "👁️ Resume Preview"
        )


        with st.expander(
            "View extracted resume text"
        ):

            st.text_area(
                "Resume Content",
                resume_text[:5000],
                height=300,
                disabled=True
            )


        # ====================================================
        # 16. PREPROCESS
        # ====================================================

        with st.spinner(
            "Preprocessing resume..."
        ):

            clean_resume = preprocess_text(
                resume_text
            )


        if not clean_resume.strip():

            st.error(
                "Not enough useful text for classification."
            )

            st.stop()


        # ====================================================
        # 17. PREDICT
        # ====================================================

        with st.spinner(
            "🤖 Analyzing resume..."
        ):

            (
                prediction,
                closest_category,
                best_score,
                score_margin,
                category_scores

            ) = predict_resume(
                clean_resume
            )


        # ====================================================
        # 18. RESULT
        # ====================================================

        st.divider()

        st.header(
            "🎯 Classification Result"
        )


        if prediction == "Other / Unknown Resume":

            st.warning(
                "⚠️ Other / Unknown Resume"
            )

            st.write(
                """
                This resume does not confidently match
                any of the four categories learned
                during model training.
                """
            )

            st.write(
                f"**Closest Trained Category:** {closest_category}"
            )

        else:

            st.success(
                f"✅ Predicted Category: {prediction}"
            )

            st.write(
                """
                The resume strongly matches one of
                the categories learned by the model.
                """
            )


        # ====================================================
        # 19. MODEL SCORES
        # ====================================================

        result1, result2 = st.columns(2)


        with result1:

            st.metric(
                "Highest Decision Score",
                f"{best_score:.3f}"
            )


        with result2:

            st.metric(
                "Top-Class Margin",
                f"{score_margin:.3f}"
            )


        # ====================================================
        # 20. CATEGORY SCORES
        # ====================================================

        st.subheader(
            "📈 Category Matching Scores"
        )


        score_dictionary = {}

        for category, score in zip(
            model.classes_,
            category_scores
        ):

            score_dictionary[
                str(category)
            ] = float(score)


        st.bar_chart(
            score_dictionary
        )


        with st.expander(
            "View category scores"
        ):

            for category, score in score_dictionary.items():

                st.write(
                    f"{category}: {score:.3f}"
                )


        # ====================================================
        # 21. EXPLANATION
        # ====================================================

        if prediction == "Other / Unknown Resume":

            st.info(
                """
                For example, a Data Scientist,
                Java Developer, Python Developer,
                Accountant or HR resume may receive
                this result because those categories
                were not included in model training.
                """
            )

        else:

            st.info(
                """
                The result was generated using
                TF-IDF text features and the trained
                Linear SVM classifier.
                """
            )


    except Exception as e:

        st.error(
            "Error while processing the resume."
        )

        st.write(
            f"Error details: {e}"
        )


# ============================================================
# 22. HOW IT WORKS
# ============================================================

st.divider()

st.header(
    "⚙️ How The System Works"
)


step1, step2, step3, step4, step5 = st.columns(5)


with step1:

    st.subheader(
        "1️⃣ Upload"
    )

    st.write(
        "Upload PDF or DOCX resume."
    )


with step2:

    st.subheader(
        "2️⃣ Extract"
    )

    st.write(
        "Extract resume text."
    )


with step3:

    st.subheader(
        "3️⃣ Process"
    )

    st.write(
        "Clean and preprocess text."
    )


with step4:

    st.subheader(
        "4️⃣ Analyze"
    )

    st.write(
        "TF-IDF and SVM analyze the resume."
    )


with step5:

    st.subheader(
        "5️⃣ Result"
    )

    st.write(
        "Return category or Unknown."
    )


# ============================================================
# 23. IMPORTANT LIMITATION
# ============================================================

st.divider()

st.header(
    "ℹ️ About Other Resumes"
)

st.write(
    """
    The current model was trained on four resume categories:
    React Developer, SQL Developer, PeopleSoft Resume and Workday.

    If another type of resume is uploaded, the system attempts
    to detect that it is outside the trained categories and
    returns **Other / Unknown Resume**.

    To predict the actual profession of many different resumes,
    the training dataset must contain those additional job
    categories.
    """
)


# ============================================================
# 24. TECHNOLOGIES
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


with tech2:

    st.subheader(
        "🧠 Linear SVM"
    )


with tech3:

    st.subheader(
        "🔤 TF-IDF"
    )


with tech4:

    st.subheader(
        "🌐 Streamlit"
    )


# ============================================================
# 25. FOOTER
# ============================================================

st.divider()

st.subheader(
    "📄 AI Resume Classification System"
)

st.caption(
    "Machine Learning and Natural Language Processing Project"
)
