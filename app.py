from pathlib import Path
import re

import joblib
import streamlit as st
import torch
from torch import nn

# ---------------- SETTINGS ----------------
MODEL_PATH = Path("models/fake_news_model.pt")
VOCAB_PATH = Path("models/vocab.pkl")

MAX_LENGTH = 300
EMBEDDING_DIM = 128
HIDDEN_DIM = 64


# ---------------- MODEL ----------------
class FakeNewsLSTM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=EMBEDDING_DIM,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=EMBEDDING_DIM,
            hidden_size=HIDDEN_DIM,
            batch_first=True,
            bidirectional=True
        )

        self.dropout = nn.Dropout(0.35)
        self.classifier = nn.Linear(HIDDEN_DIM * 2, 1)

    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)

        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]

        combined = torch.cat((forward_hidden, backward_hidden), dim=1)
        combined = self.dropout(combined)

        return self.classifier(combined).squeeze(1)


# ---------------- NLP HELPERS ----------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def encode_text(text, vocab):
    words = clean_text(text).split()[:MAX_LENGTH]

    token_ids = [
        vocab.get(word, vocab["<UNK>"])
        for word in words
    ]

    token_ids.extend([vocab["<PAD>"]] * (MAX_LENGTH - len(token_ids)))

    return token_ids


@st.cache_resource
def load_model_and_vocab():
    if not MODEL_PATH.exists() or not VOCAB_PATH.exists():
        return None, None

    vocab = joblib.load(VOCAB_PATH)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True
    )

    model = FakeNewsLSTM(checkpoint["vocab_size"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, vocab


def predict(text, model, vocab):
    encoded = encode_text(text, vocab)

    input_tensor = torch.tensor(
        [encoded],
        dtype=torch.long
    )

    with torch.no_grad():
        output = model(input_tensor)
        real_probability = torch.sigmoid(output).item()

    fake_probability = 1 - real_probability

    if real_probability >= 0.5:
        label = "Likely Real News"
        confidence = real_probability
    else:
        label = "Likely Fake News"
        confidence = fake_probability

    return label, confidence, fake_probability, real_probability


# ---------------- STREAMLIT UI ----------------
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Fake News Detection Dashboard")
st.write("Paste a news headline or article to classify it using a PyTorch NLP model.")

model, vocab = load_model_and_vocab()

if model is None:
    st.error("Model files were not found.")
    st.info("Run `python train.py` before starting the Streamlit application.")
    st.stop()

article = st.text_area(
    "News article text",
    height=250,
    placeholder="Paste a headline, news article, or social-media news claim here..."
)

if st.button("Analyze News", type="primary"):
    if not article.strip():
        st.warning("Please enter article text first.")
    else:
        label, confidence, fake_probability, real_probability = predict(
            article,
            model,
            vocab
        )

        st.subheader(label)
        st.metric("Model confidence", f"{confidence:.1%}")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Fake probability", f"{fake_probability:.1%}")

        with col2:
            st.metric("Real probability", f"{real_probability:.1%}")

        st.progress(real_probability, text=f"Probability of real news: {real_probability:.1%}")

        st.warning(
            "This is a machine-learning estimate, not factual proof. "
            "Always verify important claims using trusted sources."
        )

st.divider()
st.caption("Built with Python, PyTorch, NLP, and Streamlit.")