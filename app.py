import streamlit as st
import joblib
import nltk
import re
import string

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")


# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="News Article Categorization",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# Load CSS
# -----------------------------
def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# -----------------------------
# Load Model
# -----------------------------

model = joblib.load("news_classifier.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def preprocess(text):

    text = text.lower()

    text = re.sub(r"http\S+|www\S+", "", text)

    text = re.sub(r"<.*?>", "", text)

    text = text.translate(str.maketrans("", "", string.punctuation))

    text = re.sub(r"\d+", "", text)

    tokens = word_tokenize(text)

    tokens = [word for word in tokens if word not in stop_words]

    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)
# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="header">
    <h1>News Article Categorization</h1>
    <p>Advanced Text Analytics & NLP</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------
# Layout
# -----------------------------
left, right = st.columns([3,1])

# -----------------------------
# Left Side
# -----------------------------
with left:

    st.markdown("### Enter News Article")

    news = st.text_area(
        "",
        height=250,
        placeholder="""
Examples:

• Virat Kohli scored a brilliant century against Australia.

• Apple launched a new AI-powered smartphone.

• The stock market witnessed a record rise today.
"""
    )

    predict = st.button(
    "🔍 Predict Category",
    use_container_width=True
)


if predict:

    if news.strip() == "":

        st.warning("Please enter a news article.")

    else:

        processed = preprocess(news)

        vector = tfidf.transform([processed])

        prediction = model.predict(vector)[0]

        st.success(f"Prediction: {prediction}")



# -----------------------------
# Right Side
# -----------------------------
with right:

    st.markdown("### Project Information")

    st.info("""
Dataset : 1739 Articles

Categories : 5

Algorithm :

Multinomial Naive Bayes

Accuracy :

96.84%
""")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")

st.markdown("""
<div class="footer">

Developed by

<h3>Pratham Vernekar</h3>

M.Sc Data Science

</div>
""", unsafe_allow_html=True)