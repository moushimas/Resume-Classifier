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
# 2. CUSTOM CSS
# ============================================================

st.markdown("""
<style>

/* ---------------------------------------------------------
   MAIN APP
--------------------------------------------------------- */

.stApp {
    background-color: #f4f7fb;
}

/* Main content */
.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}


/* ---------------------------------------------------------
   MAKE TEXT CLEARLY VISIBLE
--------------------------------------------------------- */

.stApp,
.stApp p,
.stApp label,
.stApp span {
    color: #1f2937;
}

h1, h2, h3, h4 {
    color: #172554 !important;
}


/* ---------------------------------------------------------
   MAIN HEADER
--------------------------------------------------------- */

.hero {
    background: linear-gradient(
        135deg,
        #172554,
        #1d4ed8
    );

    padding: 38px 30px;

    border-radius: 18px;

    text-align: center;

    margin-bottom: 30px;

    box-shadow: 0px 8px 25px rgba(0,0,0,0.12);
}

.hero h1 {
    color: white !important;
    font-size: 42px;
    margin: 0;
}

.hero p {
    color: #e0e7ff !important;
    font-size: 18px;
    margin-top: 12px;
}


/* ---------------------------------------------------------
   CATEGORY CARDS
--------------------------------------------------------- */

.category-card {

    background-color: white;

    border: 1px solid #e2e8f0;

    border-radius: 14px;

    padding: 22px 10px;

    text-align: center;

    min-height: 125px;

    box-shadow:
        0 4px 12px rgba(0,0,0,0.06);
}

.category-icon {
    font-size: 32px;
}

.category-name {

    color: #172554 !important;

    font-size: 16px;

    font-weight: 700;

    margin-top: 8px;
}


/* ---------------------------------------------------------
   RESULT
--------------------------------------------------------- */

.result-box {

    background-color: #ecfdf5;

    border: 2px solid #10b981;

    border-radius: 16px;

    padding: 30px;

    text-align: center;

    margin-top: 20px;

    box-shadow:
        0 5px 15px rgba(0,0,0,0.07);
}

.result-label {

    color: #374151 !important;

    font-size: 17px;
}

.result-value {

    color: #047857 !important;

    font-size: 32px;

    font-weight: 800;

    margin-top: 8px;
}


/* ---------------------------------------------------------
   FILE UPLOADER
--------------------------------------------------------- */

[data-testid="stFileUploader"] {

    background-color: white;

    padding: 20px;

    border-radius: 14px;

    border: 2px dashed #94a3b8;
}


/* uploader text */

[data-testid="stFileUploader"] * {
    color: #1f2937 !important;
}


/* ---------------------------------------------------------
   METRICS
--------------------------------------------------------- */

[data-testid="stMetric"] {

    background-color: white;

    padding: 18px;

    border-radius: 12px;

    border: 1px solid #e2e8f0;
}

[data-testid="stMetricLabel"] * {

    color: #475569 !important;
}

[data-testid="stMetricValue"] * {

    color: #172554 !important;
}


/* ---------------------------------------------------------
   TEXT AREA
--------------------------------------------------------- */

textarea {

    color: #111827 !important;

    background-color: white !important;
}


/* ---------------------------------------------------------
   SIDEBAR
--------------------------------------------------------- */

[data-testid="stSidebar"] {

    background-color: #172554;
}

[data-testid="stSidebar"] * {

    color: white !important;
}


/* ---------------------------------------------------------
   HOW IT WORKS CARDS
--------------------------------------------------------- */

.step-card {

    background-color: white;

    border-radius: 14px;

    padding: 20px;

    min-height: 170px;

    border: 1px solid #e2e8f0;

    box-shadow:
        0 3px 10px rgba(0,0,0,0.05);
}

.step-card h4 {

    color: #172554 !important;
}

.step-card p {

    color: #475569 !important;
}


/* ---------------------------------------------------------
   FOOTER
--------------------------------------------------------- */

.footer {

    text-align: center;

    color: #64748b !important;

    font-size: 14px;

    padding: 30px 10px;
}

.footer b {

    color: #172554 !important;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. NLTK
# ============================================================

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


# ============================================================
# 4. LOAD TRAINED MODEL
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
# 5. TEXT PREPROCESSING
# ============================================================

def preprocess_text(text):

    # lowercase
    text = str(text).lower()

    # remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    # remove email addresses
    text = re.sub(
        r"\S+@\S+",
        " ",
        text
    )

    # remove long numbers
    text = re.sub(
        r"\d{10,}",
        " ",
        text
    )

    # remove punctuation
    text = text.translate(
        str.maketrans(
            string.punctuation,
            " " * len(string.punctuation)
        )
    )

    # split words
    words = text.split()

    # stopword removal + stemming
    words = [

        stemmer.stem(word)

        for word in words

        if word not in stop_words
        and len(word) > 1

    ]

    return " ".join(words)


# ============================================================
# 6. PDF TEXT EXTRACTION
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
# 7. DOCX TEXT EXTRACTION
# ============================================================

def extract_docx(file):

    return docx2txt.process(file)


# ============================================================
# 8. SIDEBAR
# ============================================================

with st.sidebar:

    st.title("📄 Resume AI")

    st.markdown("---")

    st.subheader("About Project")

    st.write(
        """
        AI Resume Classification System uses
        Natural Language Processing and Machine
        Learning to automatically classify resumes.
        """
    )

    st.markdown("---")

    st.subheader("Machine Learning")

    st.write("🤖 **Model:** Linear SVM")

    st.write("🔤 **Features:** TF-IDF")

    st.write("📂 **Categories:** 4")

    st.markdown("---")

    st.subheader("Supported Formats")

    st.write("📕 PDF")

    st.write("📘 DOCX")

    st.markdown("---")

    st.caption(
        "Resume Classification Project"
    )


# ============================================================
# 9. MAIN HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">

        <h1>
            📄 AI Resume Classification System
        </h1>

        <p>
            Automatically classify resumes using
            Machine Learning and Natural Language Processing
        </p>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 10. WELCOME
# ============================================================

st.subheader("👋 Welcome")

st.write(
    """
    Upload a resume and the system will analyze its
    content and predict the most suitable resume category
    using our trained **Linear SVM machine learning model**.
    """
)


# ============================================================
# 11. CATEGORIES
# ============================================================

st.subheader("🎯 Supported Resume Categories")


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.markdown(
        """
        <div class="category-card">

            <div class="category-icon">
                ⚛️
            </div>

            <div class="category-name">
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

            <div class="category-icon">
                🗄️
            </div>

            <div class="category-name">
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

            <div class="category-icon">
                🏢
            </div>

            <div class="category-name">
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

            <div class="category-icon">
                💼
            </div>

            <div class="category-name">
                Workday
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


st.write("")
st.write("")


# ============================================================
# 12. UPLOAD RESUME
# ============================================================

st.subheader("📤 Upload Your Resume")

st.write(
    "Upload your resume in **PDF or DOCX format**."
)


uploaded_file = st.file_uploader(

    "Choose a resume",

    type=[
        "pdf",
        "docx"
    ],

    help="Maximum recommended file size: 10 MB"
)


# ============================================================
# 13. WHEN FILE IS UPLOADED
# ============================================================

if uploaded_file is not None:

    st.success(
        f"✅ Successfully uploaded: {uploaded_file.name}"
    )


    try:

        # ====================================================
        # EXTRACT TEXT
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
        # CHECK TEXT
        # ====================================================

        if not resume_text.strip():

            st.error(
                """
                No readable text was found in this resume.

                Please upload a text-based PDF or DOCX file.
                """
            )

            st.stop()


        # ====================================================
        # FILE INFORMATION
        # ====================================================

        st.write("")

        st.subheader("📊 Resume Information")


        info1, info2, info3 = st.columns(3)


        with info1:

            file_extension = (
                uploaded_file.name
                .split(".")[-1]
                .upper()
            )

            st.metric(
                label="File Type",
                value=file_extension
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


        # ====================================================
        # RESUME PREVIEW
        # ====================================================

        st.write("")

        with st.expander(
            "👁️ View Extracted Resume Text",
            expanded=False
        ):

            st.text_area(

                "Resume Content",

                value=resume_text[:5000],

                height=300,

                disabled=True
            )


        # ====================================================
        # PREPROCESS
        # ====================================================

        clean_resume = preprocess_text(
            resume_text
        )


        if not clean_resume.strip():

            st.error(
                "Resume contains insufficient text for prediction."
            )

            st.stop()


        # ====================================================
        # VECTORIZE
        # ====================================================

        resume_vector = vectorizer.transform(
            [clean_resume]
        )


        # ====================================================
        # PREDICT
        # ====================================================

        with st.spinner(
            "🤖 Analyzing resume..."
        ):

            prediction = model.predict(
                resume_vector
            )[0]


        # ====================================================
        # SHOW RESULT
        # ====================================================

        st.write("")

        st.subheader("🎯 Classification Result")


        st.markdown(
            f"""
            <div class="result-box">

                <div class="result-label">

                    Predicted Resume Category

                </div>

                <div class="result-value">

                    {prediction}

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


        st.write("")


        st.info(
            """
            💡 The prediction is based on the skills,
            technologies, experience and professional
            information found in the uploaded resume.
            """
        )


    except Exception as e:

        st.error(
            f"Error processing resume: {e}"
        )


# ============================================================
# 14. HOW SYSTEM WORKS
# ============================================================

st.write("")
st.write("")

st.markdown("---")


st.subheader("⚙️ How The System Works")


step1, step2, step3, step4 = st.columns(4)


with step1:

    st.markdown(
        """
        <div class="step-card">

        <h4>
        1️⃣ Upload
        </h4>

        <p>
        Upload a resume in
        PDF or DOCX format.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


with step2:

    st.markdown(
        """
        <div class="step-card">

        <h4>
        2️⃣ Preprocessing
        </h4>

        <p>
        Resume text is extracted,
        cleaned and preprocessed.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


with step3:

    st.markdown(
        """
        <div class="step-card">

        <h4>
        3️⃣ TF-IDF
        </h4>

        <p>
        Text is converted into
        numerical features using TF-IDF.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


with step4:

    st.markdown(
        """
        <div class="step-card">

        <h4>
        4️⃣ Prediction
        </h4>

        <p>
        Linear SVM analyzes the
        features and predicts a category.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# 15. TECHNOLOGIES
# ============================================================

st.write("")
st.write("")

st.subheader("🛠️ Technologies Used")

st.write(
    """
    **Python • Streamlit • Scikit-learn • NLTK •
    TF-IDF • Linear SVM • PDFPlumber**
    """
)


# ============================================================
# 16. FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">

        <hr>

        <b>
        📄 AI Resume Classification System
        </b>

        <br><br>

        Machine Learning & Natural Language Processing Project

    </div>
    """,
    unsafe_allow_html=True
)
