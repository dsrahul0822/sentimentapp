import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK stopwords quietly
nltk.download("stopwords", quiet=True)

# ─────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Restaurant Review Sentiment Analyser",
    page_icon="🍽️",
    layout="wide",
)

# ─────────────────────────────────────────────
#  Custom CSS
# ─────────────────────────────────────────────
st.markdown(
    """
    <style>
        .main-title  { font-size:2.4rem; font-weight:700; color:#2C3E50; }
        .section-hdr { font-size:1.3rem; font-weight:600; color:#2980B9; margin-top:1rem; }
        .positive-box{
            background:#d4edda; border:1px solid #28a745;
            border-radius:10px; padding:20px; text-align:center;
        }
        .negative-box{
            background:#f8d7da; border:1px solid #dc3545;
            border-radius:10px; padding:20px; text-align:center;
        }
        .metric-box{
            background:#f0f4f8; border-radius:10px;
            padding:15px; text-align:center;
        }
        .stButton>button{
            background:#2980B9; color:white;
            border-radius:8px; padding:0.5rem 1.5rem;
            font-size:1rem; font-weight:600; border:none;
        }
        .stButton>button:hover{ background:#1a5276; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
#  Session-state keys
# ─────────────────────────────────────────────
for key in ["model_trained", "classifier", "vectorizer", "accuracy",
            "cm", "cr", "x_test", "y_test"]:
    if key not in st.session_state:
        st.session_state[key] = None
st.session_state.setdefault("model_trained", False)

# ─────────────────────────────────────────────
#  Helper: text pre-processing
# ─────────────────────────────────────────────
def preprocess(text: str) -> str:
    ps = PorterStemmer()
    all_sw = stopwords.words("english")
    all_sw.remove("not")
    review = re.sub("[^a-zA-Z]", " ", text)
    review = review.lower().split()
    review = [ps.stem(w) for w in review if w not in set(all_sw)]
    return " ".join(review)

# ─────────────────────────────────────────────
#  App header
# ─────────────────────────────────────────────
st.markdown('<p class="main-title">🍽️ Restaurant Review Sentiment Analyser</p>',
            unsafe_allow_html=True)
st.markdown("Upload your dataset, choose vectorisation, set the train/test split, "
            "train the model, and then predict new reviews — all in one place.")
st.divider()

# ═════════════════════════════════════════════
#  STEP 1 – Upload Data
# ═════════════════════════════════════════════
st.markdown('<p class="section-hdr">Step 1 — Upload Dataset</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload a **TSV** file with columns `Review` and `Liked`",
    type=["tsv", "csv"],
)

dataset = None
if uploaded_file:
    sep = "\t" if uploaded_file.name.endswith(".tsv") else ","
    dataset = pd.read_csv(uploaded_file, delimiter=sep)

    if "Review" not in dataset.columns or "Liked" not in dataset.columns:
        st.error("❌ The file must have exactly two columns: **Review** and **Liked**.")
        dataset = None
    else:
        st.success(f"✅ Loaded **{len(dataset):,}** rows.")
        with st.expander("Preview data (first 10 rows)"):
            st.dataframe(dataset.head(10), use_container_width=True)
        col1, col2 = st.columns(2)
        col1.metric("Total Reviews", len(dataset))
        col2.metric("Positive / Negative",
                    f"{dataset['Liked'].sum()} / {(dataset['Liked']==0).sum()}")

# ═════════════════════════════════════════════
#  STEP 2 – Configuration
# ═════════════════════════════════════════════
st.markdown('<p class="section-hdr">Step 2 — Configure Model</p>', unsafe_allow_html=True)

col_v, col_s, col_f = st.columns(3)

with col_v:
    vec_choice = st.selectbox(
        "Vectorisation Method",
        ["CountVectorizer (Bag of Words)", "TF-IDF Vectorizer"],
    )

with col_s:
    test_size = st.slider(
        "Test Split (%)",
        min_value=10, max_value=40, value=20, step=5,
        help="Percentage of data reserved for testing",
    )
    st.caption(f"Train: **{100-test_size}%**  |  Test: **{test_size}%**")

with col_f:
    max_features = st.number_input(
        "Max Vocabulary Features",
        min_value=100, max_value=5000, value=1500, step=100,
    )

# ═════════════════════════════════════════════
#  STEP 3 – Train Model
# ═════════════════════════════════════════════
st.markdown('<p class="section-hdr">Step 3 — Train the Model</p>', unsafe_allow_html=True)

train_btn = st.button("🚀 Train Model", disabled=(dataset is None))

if train_btn and dataset is not None:
    with st.spinner("Preprocessing text and training model…"):

        # Pre-process
        corpus = [preprocess(r) for r in dataset["Review"]]
        y = dataset["Liked"].values

        # Vectorise
        if vec_choice.startswith("CountVectorizer"):
            vec = CountVectorizer(max_features=int(max_features), binary=True)
        else:
            vec = TfidfVectorizer(max_features=int(max_features))

        X = vec.fit_transform(corpus).toarray()

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size / 100, random_state=0
        )

        # Train
        clf = BernoulliNB(alpha=1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Store in session state
        st.session_state.model_trained = True
        st.session_state.classifier   = clf
        st.session_state.vectorizer   = vec
        st.session_state.accuracy     = accuracy_score(y_test, y_pred)
        st.session_state.cm           = confusion_matrix(y_test, y_pred)
        st.session_state.cr           = classification_report(y_test, y_pred, output_dict=True)
        st.session_state.x_test       = X_test
        st.session_state.y_test       = y_test

    st.success("✅ Model trained successfully!")

# Show results if model is trained
if st.session_state.model_trained:
    st.markdown("#### 📊 Model Performance")

    m1, m2, m3, m4 = st.columns(4)
    acc = st.session_state.accuracy
    cr  = st.session_state.cr
    m1.metric("Accuracy",   f"{acc*100:.1f}%")
    m2.metric("Precision",  f"{cr['weighted avg']['precision']*100:.1f}%")
    m3.metric("Recall",     f"{cr['weighted avg']['recall']*100:.1f}%")
    m4.metric("F1 Score",   f"{cr['weighted avg']['f1-score']*100:.1f}%")

    col_cm, col_cr = st.columns(2)

    with col_cm:
        st.markdown("**Confusion Matrix**")
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(
            st.session_state.cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"], ax=ax,
        )
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig)

    with col_cr:
        st.markdown("**Classification Report**")
        report_df = pd.DataFrame(cr).T.iloc[:-1, :3].round(2)
        report_df.index = ["Negative", "Positive", "Accuracy*", "Macro avg",
                           "Weighted avg"][:len(report_df)]
        st.dataframe(report_df, use_container_width=True)

# ═════════════════════════════════════════════
#  STEP 4 – Predict New Review
# ═════════════════════════════════════════════
st.divider()
st.markdown('<p class="section-hdr">Step 4 — Predict New Feedback</p>',
            unsafe_allow_html=True)

if not st.session_state.model_trained:
    st.info("⬆️ Please upload a dataset and train the model first.")
else:
    new_review = st.text_area(
        "Enter a restaurant review:",
        placeholder="e.g. The food was absolutely delicious and the staff were very friendly!",
        height=120,
    )

    predict_btn = st.button("🔍 Predict Sentiment")

    if predict_btn:
        if not new_review.strip():
            st.warning("⚠️ Please enter a review before predicting.")
        else:
            processed = preprocess(new_review)
            vec_review = st.session_state.vectorizer.transform([processed]).toarray()
            prediction = st.session_state.classifier.predict(vec_review)[0]
            proba      = st.session_state.classifier.predict_proba(vec_review)[0]

            pos_prob = proba[1] * 100
            neg_prob = proba[0] * 100

            st.markdown("---")
            st.markdown("#### 🎯 Prediction Result")

            if prediction == 1:
                st.markdown(
                    f"""
                    <div class="positive-box">
                        <h2 style="color:#155724; margin:0">😊 POSITIVE Review</h2>
                        <p style="font-size:1.1rem; margin:8px 0 0">
                            Confidence: <strong>{pos_prob:.1f}%</strong>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="negative-box">
                        <h2 style="color:#721c24; margin:0">😞 NEGATIVE Review</h2>
                        <p style="font-size:1.1rem; margin:8px 0 0">
                            Confidence: <strong>{neg_prob:.1f}%</strong>
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("#### Probability Breakdown")
            prob_col1, prob_col2 = st.columns(2)
            prob_col1.metric("😊 Positive Probability", f"{pos_prob:.2f}%")
            prob_col2.metric("😞 Negative Probability", f"{neg_prob:.2f}%")

            # Probability bar chart
            fig2, ax2 = plt.subplots(figsize=(5, 2.5))
            bars = ax2.barh(
                ["Negative", "Positive"],
                [neg_prob, pos_prob],
                color=["#e74c3c", "#2ecc71"],
            )
            ax2.set_xlim(0, 100)
            ax2.set_xlabel("Probability (%)")
            ax2.set_title("Sentiment Probability")
            for bar, val in zip(bars, [neg_prob, pos_prob]):
                ax2.text(val + 1, bar.get_y() + bar.get_height() / 2,
                         f"{val:.1f}%", va="center", fontsize=10)
            st.pyplot(fig2)

# ─────────────────────────────────────────────
#  Footer
# ─────────────────────────────────────────────
st.divider()
st.caption("Built with Streamlit · Naive Bayes (BernoulliNB) · scikit-learn · NLTK")
