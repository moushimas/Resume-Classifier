import streamlit as st
import pdfplumber
import docx2txt
import joblib
import nltk
import re
import string

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Resume Classifier",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

/* Main background */
.stApp {
    background: linear-gradient(
        135deg,
        #f5f7fa 0%,
        #eef2f7 100%
    );
}

/* Main container */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1200px;
}


/* Header */
.main-header {
    background: linear-gradient(
        135deg,
        #1e3c72,
        #2a5298
    );

    padding: 35px;
    border-radius: 18px;

    text-align: center;

    color: white;

    margin-bottom: 25px;

    box-shadow:
        0 8px 25px rgba(0,0,0,0.12);
}


.main-header h1 {
    font-size: 42px;
    margin-bottom: 8px;
}


.main-header p {
    font-size: 17px;
    opacity: 0.9;
}


/* Cards */

.card {

    background: white;

    padding: 25px;

    border-radius: 16px;

    box-shadow:
        0 4px 18px rgba(0,0,0,0.08);

    margin-bottom: 20px;
}


/* Category cards */

.category-card {

    background: white;

    padding: 18px;

    border-radius: 12px;

    text-align: center;

    border: 1px solid #e5e7eb;

    box-shadow:
        0 3px 10px rgba(0,0,0,0.05);
}


.category-title {

    font-size: 16px;

    font-weight: 600;

    color: #1e3c72;
}


/* Prediction result */

.prediction-box {

    background: linear-gradient(
        135deg,
        #e8f5e9,
        #f1f8e9
    );

    border-left: 6px solid #2e7d32;

    padding: 25px;

    border-radius: 12px;

    margin-top: 20px;

    text-align: center;

    box-shadow:
        0 4px 15px rgba(0,0,0,0.08);
}


.prediction-title {

    font-size: 17px;

    color: #555;
}


.prediction-category {

    font-size: 30px;

    font-weight: bold;

    color: #1b5e20;

    margin-top: 8px;
}


/* Footer */

.footer {

    text-align: center;

    color: #777;

    padding-top: 30px;

    font-size: 14px;
}


/* Upload box */

[data-testid="stFileUploader"] {

    background-color: white;

    padding: 20px;

    border-radius: 12px;

    border: 2px dashed #b0bec5;
}


/* Sidebar */

[data-testid="stSidebar"] {

    background-color: #ffffff;

}

</style>
""", unsafe_allow_html=True)


# =========================================================
# NLTK
# =========================================================

try:

    stop_words = set(
        stopwords.words("english")
    )

except LookupError:

    nltk.download(
        "stopwords",
        quiet=True
    )

    stop_words = set(
        stopwords.words("english")
    )


stemmer = PorterStemmer()


# =========================================================
# LOAD MODEL
# =========================================================

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
        f"Unable to load model: {e}"
    )

    st.stop()


# =========================================================
# PREPROCESSING
# =========================================================

def preprocess_text(text):

    text = str(text).lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    # Remove email
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


# =========================================================
# PDF EXTRACTION
# =========================================================

def extract_pdf(file):

    text = ""

    with pdfplumber.open(file) as pdf:

        for page in pdf.pages:

            page_text = page.extract_text()

            if page_text:

                text += page_text + "\n"

    return text


# =========================================================
# DOCX EXTRACTION
# =========================================================

def extract_docx(file):

    return docx2txt.process(file)


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.title("📄 Resume AI")

    st.markdown("---")

    st.subheader("About")

    st.write(
        """
        This application uses Machine Learning
        and Natural Language Processing to
        classify resumes based on their
        technical content.
        """
    )

    st.markdown("---")

    st.subheader("Model")

    st.write("**Algorithm:** Linear SVM")

    st.write("**Text Features:** TF-IDF")

    st.write("**Categories:** 4")

    st.markdown("---")

    st.subheader("Supported Files")

    st.write("📕 PDF")
    st.write("📘 DOCX")

    st.markdown("---")

    st.caption(
        "Resume Classification System"
    )


# =========================================================
# HEADER
# =========================================================

st.markdown(
    """
    <div class="main-header">

        <h1>📄 AI Resume Classification System</h1>

        <p>
        Intelligent resume classification using
        Machine Learning and Natural Language Processing
        </p>

    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# INTRODUCTION
# =========================================================

st.markdown(
    """
    ### 👋 Welcome

    Upload a resume and the system will analyze its
    textual content and predict the most suitable
    category using the trained **Linear SVM model**.
    """
)


# =========================================================
# CATEGORY SECTION
# =========================================================

st.markdown("### 🎯 Supported Categories")

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.markdown(
        """
        <div class="category-card">

        <div style="font-size:32px;">⚛️</div>

        <div class="category-title">
        React Developer
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with col2:

    st.markdown(
        """
        <div class="category-card">

        <div style="font-size:32px;">🗄️</div>

        <div class="category-title">
        SQL Developer
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with col3:

    st.markdown(
        """
        <div class="category-card">

        <div style="font-size:32px;">🏢</div>

        <div class="category-title">
        Peoplesoft Resume
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


with col4:

    st.markdown(
        """
        <div class="category-card">

        <div style="font-size:32px;">💼</div>

        <div class="category-title">
        Workday
        </div>

        </div>
        """,
        unsafe_allow_html=True
    )


st.write("")
st.write("")


# =========================================================
# UPLOAD SECTION
# =========================================================

st.markdown("### 📤 Upload Resume")

st.write(
    "Upload a PDF or DOCX resume to classify it."
)


uploaded_file = st.file_uploader(

    "Drag and drop your resume here",

    type=[
        "pdf",
        "docx"
    ]
)


# =========================================================
# PROCESS FILE
# =========================================================

if uploaded_file is not None:

    st.success(
        f"✓ File uploaded successfully: {uploaded_file.name}"
    )


    try:

        # -----------------------------------------
        # Extract text
        # -----------------------------------------

        if uploaded_file.name.lower().endswith(
            ".pdf"
        ):

            resume_text = extract_pdf(
                uploaded_file
            )

        else:

            resume_text = extract_docx(
                uploaded_file
            )


        # -----------------------------------------
        # Check extracted text
        # -----------------------------------------

        if not resume_text.strip():

            st.error(
                "No readable text was found in this resume."
            )

            st.stop()


        # -----------------------------------------
        # Resume information
        # -----------------------------------------

        st.markdown("### 📊 Resume Information")


        info1, info2, info3 = st.columns(3)


        with info1:

            st.metric(
                "File Type",
                uploaded_file.name.split(".")[-1].upper()
            )


        with info2:

            st.metric(
                "Characters",
                len(resume_text)
            )


        with info3:

            st.metric(
                "Words",
                len(resume_text.split())
            )


        # -----------------------------------------
        # Resume preview
        # -----------------------------------------

        with st.expander(
            "👁️ View Extracted Resume Text"
        ):

            st.text_area(
                "Resume Content",
                resume_text[:5000],
                height=300
            )


        # -----------------------------------------
        # Preprocessing
        # -----------------------------------------

        clean_resume = preprocess_text(
            resume_text
        )


        # -----------------------------------------
        # Vectorization
        # -----------------------------------------

        resume_vector = vectorizer.transform(
            [clean_resume]
        )


        # -----------------------------------------
        # Prediction
        # -----------------------------------------

        with st.spinner(
            "🤖 AI is analyzing the resume..."
        ):

            prediction = model.predict(
                resume_vector
            )[0]


        # -----------------------------------------
        # Prediction result
        # -----------------------------------------

        st.markdown(
            f"""
            <div class="prediction-box">

                <div class="prediction-title">
                🎯 Predicted Resume Category
                </div>

                <div class="prediction-category">
                {prediction}
                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        st.write("")


        st.info(
            """
            The prediction is based on the skills,
            technologies and professional information
            found in the uploaded resume.
            """
        )


    except Exception as e:

        st.error(
            f"Unable to process the resume: {e}"
        )


# =========================================================
# HOW IT WORKS
# =========================================================

st.write("")
st.markdown("---")

st.markdown("### ⚙️ How It Works")


step1, step2, step3, step4 = st.columns(4)


with step1:

    st.markdown(
        """
        #### 1️⃣ Upload

        Upload a PDF or
        DOCX resume.
        """
    )


with step2:

    st.markdown(
        """
        #### 2️⃣ Process

        Resume text is
        extracted and cleaned.
        """
    )


with step3:

    st.markdown(
        """
        #### 3️⃣ Analyze

        TF-IDF converts the
        resume into features.
        """
    )


with step4:

    st.markdown(
        """
        #### 4️⃣ Predict

        Linear SVM predicts
        the resume category.
        """
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown(
    """
    <div class="footer">

    <hr>

    <b>AI Resume Classification System</b>

    <br>

    Built using Python • Streamlit • NLP • TF-IDF • Linear SVM

    </div>
    """,
    unsafe_allow_html=True
)
