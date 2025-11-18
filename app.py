import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import streamlit as st
from wordcloud import WordCloud
from lime.lime_text import LimeTextExplainer
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

# ============================================
# PATHS
# ============================================
MODEL_DIR = "models/distilbert_generalized"
TRAIN_ART = "artifacts"
TEST_ART = "artifacts_test"

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(page_title="Fake News Detector — DistilBERT", layout="wide")
st.title("📰 Fake News Detection — DistilBERT (Full Analytics Dashboard)")
st.caption("**0 = Fake, 1 = True** — Cross-Domain Fine-Tuned")

# ============================================
# LOAD MODEL
# ============================================
@st.cache_resource
def load_model():
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_DIR)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device

tokenizer, model, device = load_model()


# ============================================
# PREDICT FUNCTION
# ============================================
def predict_text(text: str):
    enc = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=192,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        out = model(**enc)
        probs = torch.softmax(out.logits, dim=1).cpu().numpy()[0]

    return int(np.argmax(probs)), probs


# ============================================
# WORDCLOUD
# ============================================
def safe_wordcloud(text):
    if not text or not str(text).strip():
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No text provided", ha="center", fontsize=18)
        ax.axis("off")
        return fig

    wc = WordCloud(width=900, height=400, background_color="white").generate(text)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    return fig


# ============================================
# SINGLE TEXT PREDICTION
# ============================================
st.header("🔍 Try a Single Article / Headline")

txt = st.text_area("Enter text to classify:", height=130)

if st.button("Predict"):
    if not txt.strip():
        st.warning("⚠️ Please enter some text.")
    else:
        pred, probs = predict_text(txt)
        label = "✅ TRUE (1)" if pred == 1 else "🚨 FAKE (0)"
        st.subheader(f"Prediction: **{label}**")

        fig, ax = plt.subplots(figsize=(4, 3))
        ax.bar(["Fake (0)", "True (1)"], probs, color=["#c0392b", "#27ae60"])
        ax.set_ylim(0, 1)
        st.pyplot(fig)

        # LIME
        st.markdown("### 🔍 LIME Explanation")
        explainer = LimeTextExplainer(class_names=["Fake(0)", "True(1)"])
        exp = explainer.explain_instance(
            txt,
            classifier_fn=lambda texts: np.array([predict_text(t)[1] for t in texts]),
            num_features=10,
        )
        st.components.v1.html(exp.as_html(), height=420)

        # WordCloud
        st.markdown("### ☁️ Word Cloud")
        st.pyplot(safe_wordcloud(txt))

st.markdown("---")

# ============================================
# TRAIN METRICS
# ============================================
st.header("📊 Training Metrics & Visualizations")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🧾 Classification Report (Train)")
    try:
        rep = pd.read_csv(os.path.join(TRAIN_ART, "classification_report.csv"), index_col=0)
        st.dataframe(rep, use_container_width=True)
    except:
        st.warning("⚠️ Train classification_report.csv not found")

with col2:
    st.subheader("🔵 Confusion Matrix (Train)")
    try:
        st.image(os.path.join(TRAIN_ART, "confusion_matrix.png"))
    except:
        st.warning("⚠️ Training confusion matrix not found.")

st.subheader("📉 Training Curves")

try:
    hist = pd.read_csv(os.path.join(TRAIN_ART, "training_history.csv"))
    c1, c2 = st.columns(2)

    with c1:
        st.line_chart(hist["loss"], height=260)

    with c2:
        try:
            st.image(os.path.join(TRAIN_ART, "roc_curve.png"))
        except:
            st.info("Training ROC curve not available.")
except:
    st.warning("⚠️ training_history.csv not found")

st.markdown("---")

# ============================================
# TEST METRICS
# ============================================
st.header("🌍 Cross-Domain Test Evaluation")

# Accuracy Box
try:
    accs = pd.read_csv(os.path.join(TEST_ART, "accuracies.csv"))
    test_acc = accs.loc[accs["metric"] == "test_accuracy", "value"].values
    if len(test_acc):
        st.success(f"✅ **Test Accuracy: {float(test_acc[0]):.4f}**")
except:
    st.info("Test accuracy file unavailable.")

c1, c2 = st.columns(2)

with c1:
    st.subheader("📄 Classification Report (Test)")
    try:
        rep_t = pd.read_csv(os.path.join(TEST_ART, "classification_report_test.csv"), index_col=0)
        st.dataframe(rep_t, use_container_width=True)
    except:
        try:
            with open(os.path.join(TEST_ART, "classification_report_test.txt")) as f:
                st.code(f.read())
        except:
            st.warning("⚠️ No test classification report found.")

with c2:
    st.subheader("🧊 Confusion Matrix (Test)")
    try:
        st.image(os.path.join(TEST_ART, "confusion_matrix_test.png"))
    except:
        st.warning("⚠️ confusion_matrix_test.png not found")

# ROC Test Curve
st.subheader("📈 ROC Curve (Test)")
try:
    st.image(os.path.join(TEST_ART, "roc_curve_test.png"))
except:
    st.info("ROC curve not available.")

st.markdown("---")

st.caption("✅ Built with DistilBERT, Streamlit, LIME, WordCloud, Sklearn, Matplotlib")
