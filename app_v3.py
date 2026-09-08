import streamlit as st
import joblib
import nltk
import re
import string
import numpy as np
import pandas as pd

from pathlib import Path
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity


# =========================================================
# 1. PROJECT PATH
# =========================================================

BASE_DIR = Path(__file__).resolve().parent


# =========================================================
# 2. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="News Article Categorizer",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# 3. NLTK SETUP
# =========================================================

@st.cache_resource
def setup_nltk():

    resources = [
        "punkt",
        "punkt_tab",
        "stopwords",
        "wordnet",
        "omw-1.4"
    ]

    for resource in resources:
        try:
            nltk.download(resource, quiet=True)
        except Exception:
            pass

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words("english"))

    return lemmatizer, stop_words


lemmatizer, stop_words = setup_nltk()


# =========================================================
# 4. LOAD ALL MODELS
# =========================================================

@st.cache_resource
def load_models():

    classifier_path = BASE_DIR / "news_classifier.pkl"
    tfidf_path = BASE_DIR / "tfidf_vectorizer.pkl"
    lda_vectorizer_path = BASE_DIR / "lda_vectorizer.pkl"
    lda_model_path = BASE_DIR / "lda_model.pkl"
    kmeans_path = BASE_DIR / "kmeans_model.pkl"

    required_files = {
        "news_classifier.pkl": classifier_path,
        "tfidf_vectorizer.pkl": tfidf_path,
        "lda_vectorizer.pkl": lda_vectorizer_path,
        "lda_model.pkl": lda_model_path,
        "kmeans_model.pkl": kmeans_path
    }

    missing = []

    for name, path in required_files.items():

        if not path.exists():
            missing.append(name)

    if missing:

        raise FileNotFoundError(
            "Missing model files:\n"
            + "\n".join(missing)
        )

    classifier = joblib.load(
        classifier_path
    )

    tfidf = joblib.load(
        tfidf_path
    )

    lda_vectorizer = joblib.load(
        lda_vectorizer_path
    )

    lda_model = joblib.load(
        lda_model_path
    )

    kmeans_model = joblib.load(
        kmeans_path
    )

    return (
        classifier,
        tfidf,
        lda_vectorizer,
        lda_model,
        kmeans_model
    )


try:

    (
        classifier,
        tfidf,
        lda_vectorizer,
        lda_model,
        kmeans_model
    ) = load_models()

    models_loaded = True
    model_error = None

except Exception as e:

    classifier = None
    tfidf = None
    lda_vectorizer = None
    lda_model = None
    kmeans_model = None

    models_loaded = False
    model_error = str(e)


# =========================================================
# 5. LOAD NEWS DATASET
# =========================================================

@st.cache_data
def load_news_dataset():

    dataset_path = BASE_DIR / "news_articles.csv"

    if not dataset_path.exists():

        return None

    try:

        data = pd.read_csv(
            dataset_path
        )

        # Required columns
        required = [
            "description",
            "category",
            "processed_text"
        ]

        for column in required:

            if column not in data.columns:

                return None

        data["description"] = (
            data["description"]
            .fillna("")
            .astype(str)
        )

        data["category"] = (
            data["category"]
            .fillna("Unknown")
            .astype(str)
        )

        data["processed_text"] = (
            data["processed_text"]
            .fillna("")
            .astype(str)
        )

        return data

    except Exception:

        return None


news_df = load_news_dataset()


# =========================================================
# 6. TEXT PREPROCESSING
# =========================================================

def preprocess_text(text):

    text = str(text)

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        "",
        text
    )

    # Remove HTML
    text = re.sub(
        r"<.*?>",
        "",
        text
    )

    # Remove punctuation
    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    # Remove numbers
    text = re.sub(
        r"\d+",
        "",
        text
    )

    # Tokenization
    try:

        tokens = word_tokenize(
            text
        )

    except Exception:

        tokens = text.split()

    # Stopword removal
    tokens = [
        word
        for word in tokens
        if word not in stop_words
    ]

    # Lemmatization
    tokens = [
        lemmatizer.lemmatize(
            word
        )
        for word in tokens
    ]

    return " ".join(tokens)


# =========================================================
# 7. KEYWORD EXTRACTION
# =========================================================

def extract_keywords(
    text,
    top_n=10
):

    if tfidf is None:

        return []

    processed = preprocess_text(
        text
    )

    if not processed.strip():

        return []

    vector = tfidf.transform(
        [processed]
    )

    feature_names = (
        tfidf.get_feature_names_out()
    )

    scores = vector.toarray()[0]

    top_indices = (
        scores.argsort()[-top_n:][::-1]
    )

    results = []

    for index in top_indices:

        score = float(
            scores[index]
        )

        if score > 0:

            results.append(
                (
                    feature_names[index],
                    score
                )
            )

    return results


# =========================================================
# 8. GET TOP WORDS FOR LDA TOPIC
# =========================================================

def get_lda_topic_words(
    topic_index,
    number=10
):

    if lda_model is None:

        return []

    if lda_vectorizer is None:

        return []

    feature_names = (
        lda_vectorizer
        .get_feature_names_out()
    )

    topic = (
        lda_model
        .components_[topic_index]
    )

    top_indices = (
        topic.argsort()[-number:][::-1]
    )

    return [
        feature_names[index]
        for index in top_indices
    ]


# =========================================================
# 9. SIDEBAR
# =========================================================

with st.sidebar:

    st.title("📰 News Analyzer")

    st.caption(
        "Advanced Text Analytics & NLP"
    )

    st.divider()

    st.subheader("🧭 Navigation")

    page = st.radio(
        "Select Module",
        [
            "🏠 Dashboard",
            "🔍 Article Analysis",
            "🔗 Similar Articles",
            "🧠 Topic Modeling",
            "📊 K-Means Clustering",
            "🔑 Keyword Extraction",
            "📚 Project Information"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.subheader("🧠 NLP Techniques")

    st.write("✅ Text Preprocessing")
    st.write("✅ TF-IDF")
    st.write("✅ Classification")
    st.write("✅ Cosine Similarity")
    st.write("✅ LDA Topic Modeling")
    st.write("✅ K-Means Clustering")
    st.write("✅ Keyword Extraction")


# =========================================================
# 10. MAIN HEADER
# =========================================================

st.title(
    "📰News Article Categorizer"
)

st.caption(
    "Advanced Text Analytics & Natural Language Processing"
)

st.divider()


# =========================================================
# 11. DASHBOARD
# =========================================================

if page == "🏠 Dashboard":

    st.header(
        "📊 Project Dashboard"
    )

    st.write(
        "An NLP-powered system for analyzing, "
        "classifying, comparing and discovering "
        "patterns in news articles."
    )

    st.write("")

    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "📰 News Articles",
            "1,739"
        )

    with col2:

        st.metric(
            "🏷️ Categories",
            "5"
        )

    with col3:

        st.metric(
            "🎯 Classification Accuracy",
            "96.84%"
        )

    with col4:

        st.metric(
            "🧠 NLP Techniques",
            "7"
        )

    st.divider()

    # -----------------------------------------------------
    # PROJECT DESCRIPTION
    # -----------------------------------------------------

    left, right = st.columns(2)

    with left:

        st.subheader(
            "🚀 What does this project do?"
        )

        st.info(
            """
The Intelligent News Article Analyzer applies
Natural Language Processing and Machine Learning
techniques to news articles.

The system can:

• Predict the category of an article  
• Find similar articles  
• Discover hidden topics  
• Group articles into clusters  
• Extract important keywords
"""
        )

    with right:

        st.subheader(
            "🧠 Techniques Used"
        )

        st.write(
            "1. Text Preprocessing"
        )

        st.write(
            "2. TF-IDF Feature Engineering"
        )

        st.write(
            "3. Multinomial Naive Bayes Classification"
        )

        st.write(
            "4. Cosine Similarity"
        )

        st.write(
            "5. LDA Topic Modeling"
        )

        st.write(
            "6. K-Means Clustering"
        )

        st.write(
            "7. TF-IDF Keyword Extraction"
        )

    st.divider()

    # -----------------------------------------------------
    # WORKFLOW
    # -----------------------------------------------------

    st.subheader(
        "🔄 Project Workflow"
    )

    st.write(
        """
**News Article**
→
**Text Preprocessing**
→
**TF-IDF**
→
**Classification / Similarity**
→
**LDA Topics**
→
**K-Means Clusters**
→
**Keyword Extraction**
"""
    )


# =========================================================
# 12. ARTICLE ANALYSIS
# =========================================================

elif page == "🔍 Article Analysis":

    st.header(
        "🔍 News Article Classification"
    )

    st.info(
        """
Enter a news article below. The trained
Multinomial Naive Bayes model will predict
its category.
"""
    )

    article = st.text_area(
        "📝 Enter News Article",
        height=250,
        placeholder=(
            "Example:\n\n"
            "Manchester United secured an important "
            "victory after defeating their opponents "
            "in a dramatic football match..."
        )
    )

    analyze_button = st.button(
        "🔎 Analyze Article",
        type="primary",
        use_container_width=True
    )

    if analyze_button:

        if not article.strip():

            st.warning(
                "⚠️ Please enter a news article."
            )

        elif not models_loaded:

            st.error(
                "❌ Required ML models could not "
                "be loaded."
            )

            st.code(
                model_error
            )

        else:

            with st.spinner(
                "Analyzing article..."
            ):

                processed = preprocess_text(
                    article
                )

                vector = tfidf.transform(
                    [processed]
                )

                prediction = classifier.predict(
                    vector
                )[0]

                confidence = None

                if hasattr(
                    classifier,
                    "predict_proba"
                ):

                    probabilities = (
                        classifier
                        .predict_proba(
                            vector
                        )[0]
                    )

                    confidence = (
                        float(
                            np.max(
                                probabilities
                            )
                        ) * 100
                    )

            st.success(
                "✅ Analysis completed!"
            )

            st.subheader(
                "📌 Prediction Results"
            )

            c1, c2, c3 = st.columns(3)

            with c1:

                st.metric(
                    "🏷️ Predicted Category",
                    str(prediction)
                )

            with c2:

                if confidence is not None:

                    st.metric(
                        "🎯 Confidence",
                        f"{confidence:.2f}%"
                    )

                else:

                    st.metric(
                        "🎯 Confidence",
                        "N/A"
                    )

            with c3:

                st.metric(
                    "📝 Processed Words",
                    len(
                        processed.split()
                    )
                )

            if confidence is not None:

                st.subheader(
                    "📈 Model Confidence"
                )

                st.progress(
                    min(
                        confidence / 100,
                        1.0
                    )
                )

            st.divider()

            st.subheader(
                "🔑 Important Terms"
            )

            keywords = extract_keywords(
                article,
                top_n=8
            )

            if keywords:

                keyword_cols = st.columns(
                    4
                )

                for i, (
                    word,
                    score
                ) in enumerate(
                    keywords
                ):

                    with keyword_cols[
                        i % 4
                    ]:

                        st.info(
                            f"**{word}**\n\n"
                            f"TF-IDF: {score:.4f}"
                        )

            else:

                st.write(
                    "No significant keywords found."
                )

            st.divider()

            st.subheader(
                "🧹 Processed Text"
            )

            with st.expander(
                "View preprocessing result"
            ):

                st.write(
                    processed
                )


# =========================================================
# 13. COSINE SIMILARITY
# =========================================================

elif page == "🔗 Similar Articles":

    st.header(
        "🔗 Document Similarity Analyzer"
    )

    st.info(
        """
TF-IDF converts the article into a numerical
vector. Cosine Similarity then compares that
vector with the news article corpus.
"""
    )

    if news_df is None:

        st.error(
            "❌ news_articles.csv was not found."
        )

    elif not models_loaded:

        st.error(
            "❌ TF-IDF model could not be loaded."
        )

    else:

        query = st.text_area(
            "📝 Enter Article or News Headline",
            height=200,
            placeholder=(
                "Example:\n\n"
                "The football team secured a dramatic "
                "victory after scoring in the final minutes..."
            )
        )

        similarity_button = st.button(
            "🔗 Find Similar Articles",
            type="primary",
            use_container_width=True
        )

        if similarity_button:

            if not query.strip():

                st.warning(
                    "⚠️ Please enter some text."
                )

            else:

                with st.spinner(
                    "🔎 Calculating similarity..."
                ):

                    query_processed = (
                        preprocess_text(
                            query
                        )
                    )

                    query_vector = (
                        tfidf.transform(
                            [query_processed]
                        )
                    )

                    article_vectors = (
                        tfidf.transform(
                            news_df[
                                "processed_text"
                            ]
                        )
                    )

                    scores = cosine_similarity(
                        query_vector,
                        article_vectors
                    )[0]

                    top_indices = (
                        scores
                        .argsort()[::-1][:5]
                    )

                st.success(
                    "✅ Similarity analysis completed!"
                )

                st.subheader(
                    "🏆 Top 5 Similar Articles"
                )

                for rank, index in enumerate(
                    top_indices,
                    start=1
                ):

                    article_text = (
                        news_df.iloc[index][
                            "description"
                        ]
                    )

                    category = (
                        news_df.iloc[index][
                            "category"
                        ]
                    )

                    score = (
                        float(
                            scores[index]
                        ) * 100
                    )

                    if rank == 1:

                        title = (
                            "🥇 #1 Most Similar"
                        )

                    elif rank == 2:

                        title = (
                            "🥈 #2 Most Similar"
                        )

                    elif rank == 3:

                        title = (
                            "🥉 #3 Most Similar"
                        )

                    else:

                        title = (
                            f"📄 #{rank} "
                            "Most Similar"
                        )

                    with st.container(
                        border=True
                    ):

                        col1, col2 = (
                            st.columns(
                                [3, 1]
                            )
                        )

                        with col1:

                            st.subheader(
                                title
                            )

                            st.write(
                                f"🏷️ **Category:** "
                                f"{category}"
                            )

                        with col2:

                            st.metric(
                                "Similarity",
                                f"{score:.2f}%"
                            )

                        if len(
                            str(article_text)
                        ) > 500:

                            preview = (
                                str(
                                    article_text
                                )[:500]
                                + "..."
                            )

                        else:

                            preview = str(
                                article_text
                            )

                        st.write(
                            preview
                        )

                        st.progress(
                            min(
                                max(
                                    score / 100,
                                    0.0
                                ),
                                1.0
                            )
                        )

                st.divider()

                st.subheader(
                    "📊 Similarity Summary"
                )

                c1, c2, c3 = st.columns(3)

                highest = (
                    scores[
                        top_indices[0]
                    ] * 100
                )

                average = (
                    np.mean(
                        scores[
                            top_indices
                        ]
                    ) * 100
                )

                lowest = (
                    scores[
                        top_indices[-1]
                    ] * 100
                )

                with c1:

                    st.metric(
                        "🥇 Highest",
                        f"{highest:.2f}%"
                    )

                with c2:

                    st.metric(
                        "📊 Top 5 Average",
                        f"{average:.2f}%"
                    )

                with c3:

                    st.metric(
                        "5️⃣ Lowest",
                        f"{lowest:.2f}%"
                    )

                st.subheader(
                    "📈 Similarity Comparison"
                )

                chart = pd.DataFrame(
                    {
                        "Article": [
                            f"Article {i}"
                            for i in range(1, 6)
                        ],
                        "Similarity (%)": [
                            round(
                                scores[index] * 100,
                                2
                            )
                            for index in top_indices
                        ]
                    }
                )

                st.bar_chart(
                    chart.set_index(
                        "Article"
                    )
                )

                with st.expander(
                    "🧠 How Cosine Similarity works"
                ):

                    st.write(
                        """
The query article is first converted
into a TF-IDF vector.

Cosine Similarity then measures the
angle between the query vector and
each article vector.

A score closer to 1 indicates stronger
similarity, while a score closer to 0
indicates weaker similarity.

The five highest-scoring articles are
displayed as the final result.
"""
                    )


# =========================================================
# 14. LDA TOPIC MODELING
# =========================================================

elif page == "🧠 Topic Modeling":

    st.header(
        "🧠 LDA Topic Modeling"
    )

    st.info(
        """
Latent Dirichlet Allocation (LDA) is an
unsupervised topic modeling technique that
discovers hidden topics within the news corpus.
"""
    )

    if not models_loaded:

        st.error(
            "❌ LDA model could not be loaded."
        )

        st.code(
            model_error
        )

    else:

        number_of_topics = (
            lda_model.n_components
        )

        st.metric(
            "🧠 Discovered Topics",
            number_of_topics
        )

        st.divider()

        st.subheader(
            "🔎 Discovered Topics"
        )

        for topic_number in range(
            number_of_topics
        ):

            words = get_lda_topic_words(
                topic_number,
                10
            )

            st.markdown(
                f"### Topic {topic_number + 1}"
            )

            st.write(
                " • ".join(words)
            )

            # Topic word columns
            word_cols = st.columns(
                min(
                    len(words),
                    5
                )
            )

            for i, word in enumerate(
                words
            ):

                with word_cols[
                    i % len(word_cols)
                ]:

                    st.info(
                        word
                    )

            st.divider()

        # -------------------------------------------------
        # TEST A NEW ARTICLE
        # -------------------------------------------------

        st.subheader(
            "🔍 Identify Topic of an Article"
        )

        topic_article = st.text_area(
            "Enter an article",
            height=180,
            placeholder=(
                "Paste an article here to "
                "identify its dominant LDA topic..."
            )
        )

        topic_button = st.button(
            "🧠 Identify Topic",
            type="primary"
        )

        if topic_button:

            if not topic_article.strip():

                st.warning(
                    "Please enter an article."
                )

            else:

                processed = (
                    preprocess_text(
                        topic_article
                    )
                )

                lda_vector = (
                    lda_vectorizer.transform(
                        [processed]
                    )
                )

                topic_distribution = (
                    lda_model.transform(
                        lda_vector
                    )[0]
                )

                dominant_topic = int(
                    np.argmax(
                        topic_distribution
                    )
                )

                topic_probability = (
                    topic_distribution[
                        dominant_topic
                    ] * 100
                )

                st.success(
                    f"Dominant Topic: "
                    f"Topic {dominant_topic + 1}"
                )

                st.metric(
                    "Topic Probability",
                    f"{topic_probability:.2f}%"
                )

                st.subheader(
                    "📊 Topic Distribution"
                )

                topic_chart = pd.DataFrame(
                    {
                        "Topic": [
                            f"Topic {i + 1}"
                            for i in range(
                                number_of_topics
                            )
                        ],
                        "Probability": [
                            round(
                                value * 100,
                                2
                            )
                            for value in
                            topic_distribution
                        ]
                    }
                )

                st.bar_chart(
                    topic_chart.set_index(
                        "Topic"
                    )
                )


# =========================================================
# 15. K-MEANS CLUSTERING
# =========================================================

elif page == "📊 K-Means Clustering":

    st.header(
        "📊 K-Means Article Clustering"
    )

    st.info(
        """
K-Means groups articles according to similarity
in their TF-IDF feature space. Since it is an
unsupervised algorithm, the clusters are created
without using the category labels.
"""
    )

    if news_df is None:

        st.error(
            "❌ news_articles.csv was not found."
        )

    elif not models_loaded:

        st.error(
            "❌ K-Means model could not be loaded."
        )

    else:

        number_of_clusters = (
            kmeans_model.n_clusters
        )

        st.metric(
            "📊 Number of Clusters",
            number_of_clusters
        )

        # -------------------------------------------------
        # CLUSTER DISTRIBUTION
        # -------------------------------------------------

        if "cluster" in news_df.columns:

            cluster_counts = (
                news_df[
                    "cluster"
                ]
                .value_counts()
                .sort_index()
            )

        else:

            # If cluster column is missing,
            # calculate it using K-Means.
            article_vectors = (
                tfidf.transform(
                    news_df[
                        "processed_text"
                    ]
                )
            )

            cluster_labels = (
                kmeans_model.predict(
                    article_vectors
                )
            )

            cluster_counts = (
                pd.Series(
                    cluster_labels
                )
                .value_counts()
                .sort_index()
            )

        st.subheader(
            "📈 Cluster Distribution"
        )

        cluster_chart = pd.DataFrame(
            {
                "Cluster": [
                    f"Cluster {i}"
                    for i in cluster_counts.index
                ],
                "Articles": [
                    int(value)
                    for value in cluster_counts.values
                ]
            }
        )

        st.bar_chart(
            cluster_chart.set_index(
                "Cluster"
            )
        )

        st.divider()

        # -------------------------------------------------
        # CLUSTER DETAILS
        # -------------------------------------------------

        st.subheader(
            "🔎 Cluster Analysis"
        )

        for cluster_id in range(
            number_of_clusters
        ):

            cluster_size = int(
                cluster_counts.get(
                    cluster_id,
                    0
                )
            )

            st.markdown(
                f"### 📁 Cluster {cluster_id}"
            )

            st.write(
                f"**Articles:** "
                f"{cluster_size}"
            )

            # Get important words from centroid
            centroid = (
                kmeans_model
                .cluster_centers_[
                    cluster_id
                ]
            )

            feature_names = (
                tfidf
                .get_feature_names_out()
            )

            top_indices = (
                centroid
                .argsort()[-10:][::-1]
            )

            top_words = [
                feature_names[index]
                for index in top_indices
            ]

            st.write(
                "**Top Terms:** "
                + " • ".join(
                    top_words
                )
            )

            # Show sample articles
            if "cluster" in news_df.columns:

                samples = news_df[
                    news_df["cluster"]
                    == cluster_id
                ].head(3)

            else:

                samples = pd.DataFrame()

            if not samples.empty:

                with st.expander(
                    "View sample articles"
                ):

                    for _, row in samples.iterrows():

                        st.write(
                            "• "
                            + str(
                                row[
                                    "description"
                                ]
                            )[:250]
                            + "..."
                        )

            st.divider()

        # -------------------------------------------------
        # ASSIGN NEW ARTICLE TO CLUSTER
        # -------------------------------------------------

        st.subheader(
            "🔍 Assign New Article to a Cluster"
        )

        cluster_article = st.text_area(
            "Enter article",
            height=180,
            placeholder=(
                "Paste an article to find "
                "which cluster it belongs to..."
            )
        )

        cluster_button = st.button(
            "📊 Find Cluster",
            type="primary"
        )

        if cluster_button:

            if not cluster_article.strip():

                st.warning(
                    "Please enter an article."
                )

            else:

                processed = (
                    preprocess_text(
                        cluster_article
                    )
                )

                vector = tfidf.transform(
                    [processed]
                )

                assigned_cluster = (
                    kmeans_model.predict(
                        vector
                    )[0]
                )

                st.success(
                    f"This article belongs to "
                    f"**Cluster {assigned_cluster}**."
                )


# =========================================================
# 16. KEYWORD EXTRACTION
# =========================================================

elif page == "🔑 Keyword Extraction":

    st.header(
        "🔑 TF-IDF Keyword Extraction"
    )

    st.info(
        """
TF-IDF scores are used to identify terms that
are particularly important to the entered article.
"""
    )

    keyword_article = st.text_area(
        "📝 Enter Article",
        height=250,
        placeholder=(
            "Paste a news article here..."
        )
    )

    keyword_button = st.button(
        "🔑 Extract Keywords",
        type="primary",
        use_container_width=True
    )

    if keyword_button:

        if not keyword_article.strip():

            st.warning(
                "⚠️ Please enter an article."
            )

        elif not models_loaded:

            st.error(
                "❌ TF-IDF vectorizer could not be loaded."
            )

        else:

            keywords = extract_keywords(
                keyword_article,
                top_n=10
            )

            if not keywords:

                st.warning(
                    "No significant keywords "
                    "were found."
                )

            else:

                st.success(
                    f"✅ Extracted "
                    f"{len(keywords)} keywords."
                )

                st.subheader(
                    "🔑 Important Keywords"
                )

                # Display keyword metrics
                for i in range(
                    0,
                    len(keywords),
                    2
                ):

                    cols = st.columns(2)

                    for j in range(2):

                        index = i + j

                        if index < len(
                            keywords
                        ):

                            word, score = (
                                keywords[index]
                            )

                            with cols[j]:

                                st.metric(
                                    f"#{index + 1} "
                                    f"{word}",
                                    f"{score:.4f}"
                                )

                st.divider()

                # Keyword table
                st.subheader(
                    "📋 Keyword Scores"
                )

                keyword_table = (
                    pd.DataFrame(
                        keywords,
                        columns=[
                            "Keyword",
                            "TF-IDF Score"
                        ]
                    )
                )

                keyword_table[
                    "TF-IDF Score"
                ] = keyword_table[
                    "TF-IDF Score"
                ].round(4)

                st.dataframe(
                    keyword_table,
                    use_container_width=True,
                    hide_index=True
                )

                # Keyword chart
                st.subheader(
                    "📊 Keyword Importance"
                )

                chart = keyword_table.set_index(
                    "Keyword"
                )

                st.bar_chart(
                    chart
                )


# =========================================================
# 17. PROJECT INFORMATION
# =========================================================

elif page == "📚 Project Information":

    st.header(
        "📚 Project Information"
    )

    st.subheader(
        "📰 Intelligent News Article Analyzer"
    )

    st.write(
        """
This project applies Advanced Text Analytics
and Natural Language Processing techniques to
analyze a collection of news articles.
"""
    )

    st.divider()

    # -----------------------------------------------------
    # DATASET
    # -----------------------------------------------------

    st.subheader(
        "📊 Dataset"
    )

    st.write(
        "• 1,739 articles used in the classification pipeline"
    )

    st.write(
        "• 5 news categories"
    )

    st.write(
        "• BBC News dataset"
    )

    st.divider()

    # -----------------------------------------------------
    # TECHNIQUES
    # -----------------------------------------------------

    st.subheader(
        "🧠 NLP Techniques"
    )

    technique_table = pd.DataFrame(
        {
            "Technique": [
                "Text Preprocessing",
                "TF-IDF",
                "Classification",
                "Cosine Similarity",
                "LDA Topic Modeling",
                "K-Means Clustering",
                "Keyword Extraction"
            ],
            "Purpose": [
                "Clean and normalize text",
                "Convert text into numerical features",
                "Predict news categories",
                "Find similar documents",
                "Discover hidden topics",
                "Group similar articles",
                "Identify important terms"
            ]
        }
    )

    st.dataframe(
        technique_table,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # -----------------------------------------------------
    # MACHINE LEARNING
    # -----------------------------------------------------

    st.subheader(
        "🤖 Machine Learning"
    )

    st.write(
        "Algorithm: Multinomial Naive Bayes"
    )

    st.write(
        "Feature Engineering: TF-IDF"
    )

    st.write(
        "Classification Accuracy: 96.84%"
    )

    st.divider()

    # -----------------------------------------------------
    # PROJECT WORKFLOW
    # -----------------------------------------------------

    st.subheader(
        "🔄 Complete Workflow"
    )

    st.write(
        """
**1. Dataset Collection**

↓

**2. Text Preprocessing**

↓

**3. TF-IDF Feature Engineering**

↓

**4. Multinomial Naive Bayes Classification**

↓

**5. Cosine Similarity**

↓

**6. LDA Topic Modeling**

↓

**7. K-Means Clustering**

↓

**8. TF-IDF Keyword Extraction**

↓

**9. Interactive Streamlit Application**
"""
    )


# =========================================================
# 18. FOOTER
# =========================================================

st.divider()

st.caption(
    "Developed by Pratham Vernekar | M.Sc. Data Science"
)