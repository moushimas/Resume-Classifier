# ============================================================
# AI RESUME CLASSIFICATION SYSTEM
# app.py
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
# 2. NLTK STOPWORDS
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
# 4. PREPROCESS RESUME TEXT
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

    # Split into words
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
# 5. EXTRACT TEXT FROM PDF
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
# 6. EXTRACT TEXT FROM DOCX
# ============================================================

def extract_docx(file):

    text = docx2txt.process(file)

    return text


# ============================================================
# 7. SIDEBAR
# ============================================================

with st.sidebar:

    st.title("📄 Resume AI")

    st.write(
        "AI-powered Resume Classification System"
    )

    st.divider()

    st.subheader("About Project")

    st.write(
        """
        This application automatically analyzes
        resumes and predicts their category using
        Natural Language Processing and Machine Learning.
        """
    )

    st.divider()

    st.subheader("Model Information")

    st.write("**Algorithm:** Linear SVM")

    st.write("**Feature Extraction:** TF-IDF")

    st.write("**Number of Categories:** 4")

    st.divider()

    st.subheader("Supported Files")

    st.write("📕 PDF")

    st.write("📘 DOCX")

    st.divider()

    st.write(
        "Machine Learning & NLP Project"
    )


# ============================================================
# 8. MAIN TITLE
# ============================================================

st.title(
    "📄 AI Resume Classification System"
)

st.write(
    """
    Upload a resume and the system will automatically
    analyze its content and predict the most suitable
    resume category.
    """
)

st.divider()


# ============================================================
# 9. SUPPORTED CATEGORIES
# ============================================================

st.header(
    "🎯 Supported Resume Categories"
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.subheader(
        "⚛️ React Developer"
    )

    st.write(
        "Frontend and React-based development resumes."
    )


with col2:

    st.subheader(
        "🗄️ SQL Developer"
    )

    st.write(
        "Database and SQL development resumes."
    )


with col3:

    st.subheader(
        "🏢 Peoplesoft"
    )

    st.write(
        "PeopleSoft and enterprise application resumes."
    )


with col4:

    st.subheader(
        "💼 Workday"
    )

    st.write(
        "Workday and HR technology resumes."
    )


st.divider()


# ============================================================
# 10. UPLOAD RESUME
# ============================================================

st.header(
    "📤 Upload Resume"
)

st.write(
    "Select a PDF or DOCX resume from your computer."
)


uploaded_file = st.file_uploader(
    "Choose Resume",
    type=[
        "pdf",
        "docx"
    ]
)


# ============================================================
# 11. PROCESS UPLOADED RESUME
# ============================================================

if uploaded_file is not None:

    st.success(
        "Resume uploaded successfully."
    )

    st.write(
        f"**File Name:** {uploaded_file.name}"
    )


    try:

        # ====================================================
        # 12. EXTRACT TEXT
        # ====================================================

        if uploaded_file.name.lower().endswith(".pdf"):

            resume_text = extract_pdf(
                uploaded_file
            )


        elif uploaded_file.name.lower().endswith(".docx"):

            resume_text = extract_docx(
                uploaded_file
            )


        else:

            st.error(
                "Unsupported file format."
            )

            st.stop()


        # ====================================================
        # 13. CHECK EXTRACTED TEXT
        # ====================================================

        if not resume_text.strip():

            st.error(
                "No readable text was found in the resume."
            )

            st.stop()


        # ====================================================
        # 14. RESUME INFORMATION
        # ====================================================

        st.divider()

        st.header(
            "📊 Resume Information"
        )


        info1, info2, info3 = st.columns(3)


        # File Type
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


        # Word Count
        with info2:

            total_words = len(
                resume_text.split()
            )

            st.metric(
                "Total Words",
                total_words
            )


        # Character Count
        with info3:

            total_characters = len(
                resume_text
            )

            st.metric(
                "Characters",
                total_characters
            )


        # ====================================================
        # 15. RESUME PREVIEW
        # ====================================================

        st.subheader(
            "👁️ Resume Preview"
        )


        with st.expander(
            "View Extracted Resume Text"
        ):

            st.text_area(
                "Resume Content",
                value=resume_text[:5000],
                height=300,
                disabled=True
            )


        # ====================================================
        # 16. PREPROCESS RESUME
        # ====================================================

        clean_resume = preprocess_text(
            resume_text
        )


        if not clean_resume.strip():

            st.error(
                "The resume does not contain enough text for prediction."
            )

            st.stop()


        # ====================================================
        # 17. TF-IDF TRANSFORMATION
        # ====================================================

        resume_vector = vectorizer.transform(
            [clean_resume]
        )


        # ====================================================
        # 18. MODEL PREDICTION
        # ====================================================

        st.divider()

        st.header(
            "🤖 Resume Analysis"
        )


        with st.spinner(
            "Analyzing resume..."
        ):

            prediction = model.predict(
                resume_vector
            )[0]


        # ====================================================
        # 19. DISPLAY RESULT
        # ====================================================

        st.success(
            "Resume analysis completed successfully."
        )


        st.subheader(
            "🎯 Predicted Category"
        )


        st.title(
            prediction
        )


        st.info(
            """
            The predicted category is determined from
            the skills, technologies, experience and
            professional information available in the resume.
            """
        )


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as e:

        st.error(
            f"Error processing resume: {e}"
        )


# ============================================================
# 20. HOW THE SYSTEM WORKS
# ============================================================

st.divider()

st.header(
    "⚙️ How the System Works"
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
        Resume text is extracted
        and cleaned using NLP.
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
# 21. TECHNOLOGIES
# ============================================================

st.divider()

st.header(
    "🛠️ Technologies Used"
)


tech1, tech2, tech3 = st.columns(3)


with tech1:

    st.subheader(
        "Programming"
    )

    st.write("🐍 Python")

    st.write("🎈 Streamlit")


with tech2:

    st.subheader(
        "Machine Learning"
    )

    st.write("🤖 Linear SVM")

    st.write("🔤 TF-IDF")

    st.write("📊 Scikit-learn")


with tech3:

    st.subheader(
        "Text Processing"
    )

    st.write("📚 NLTK")

    st.write("📄 PDFPlumber")

    st.write("📘 Docx2txt")


# ============================================================
# 22. FOOTER
# ============================================================

st.divider()

st.subheader(
    "📄 AI Resume Classification System"
)

st.write(
    "Machine Learning and Natural Language Processing Project"
)
