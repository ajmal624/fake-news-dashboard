from collections import Counter
from pathlib import Path
import re

import joblib
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset

# ---------------- SETTINGS ----------------
RAW_DATA = Path("data/raw")
MODEL_DIR = Path("models")

MAX_VOCAB_SIZE = 20000
MAX_LENGTH = 300
EMBEDDING_DIM = 128
HIDDEN_DIM = 64
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001
RANDOM_STATE = 42

# ---------------- TEXT PROCESSING ----------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def combine_article_text(dataframe):
    title = dataframe["title"].fillna("") if "title" in dataframe.columns else ""
    text = dataframe["text"].fillna("") if "text" in dataframe.columns else ""
    return (title + " " + text).apply(clean_text)


def build_vocab(texts):
    word_counter = Counter()

    for text in texts:
        word_counter.update(text.split())

    vocab = {
        "<PAD>": 0,
        "<UNK>": 1
    }

    for index, (word, _) in enumerate(
        word_counter.most_common(MAX_VOCAB_SIZE - 2),
        start=2
    ):
        vocab[word] = index

    return vocab


def encode_text(text, vocab):
    words = text.split()[:MAX_LENGTH]
    token_ids = [vocab.get(word, vocab["<UNK>"]) for word in words]

    padding_needed = MAX_LENGTH - len(token_ids)
    token_ids.extend([vocab["<PAD>"]] * padding_needed)

    return token_ids


# ---------------- DATASET ----------------
class NewsDataset(Dataset):
    def __init__(self, texts, labels, vocab):
        self.features = torch.tensor(
            [encode_text(text, vocab) for text in texts],
            dtype=torch.long
        )
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]


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


# ---------------- TRAINING ----------------
def main():
    fake_path = RAW_DATA / "Fake.csv"
    real_path = RAW_DATA / "True.csv"

    if not fake_path.exists() or not real_path.exists():
        print("Dataset files were not found.")
        print("Place Fake.csv and True.csv in data/raw/")
        return

    print("Loading dataset...")

    fake_data = pd.read_csv(fake_path)
    real_data = pd.read_csv(real_path)

    fake_data = pd.DataFrame({
        "text": combine_article_text(fake_data),
        "label": 0
    })

    real_data = pd.DataFrame({
        "text": combine_article_text(real_data),
        "label": 1
    })

    data = pd.concat([fake_data, real_data], ignore_index=True)
    data = data[data["text"].str.len() > 20]
    data = data.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    train_data, test_data = train_test_split(
        data,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=data["label"]
    )

    train_texts = train_data["text"].tolist()
    train_labels = train_data["label"].tolist()

    test_texts = test_data["text"].tolist()
    test_labels = test_data["label"].tolist()

    print("Building vocabulary...")
    vocab = build_vocab(train_texts)

    train_dataset = NewsDataset(train_texts, train_labels, vocab)
    test_dataset = NewsDataset(test_texts, test_labels, vocab)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = FakeNewsLSTM(len(vocab)).to(device)

    loss_function = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_function(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        average_loss = running_loss / len(train_loader)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"- Training Loss: {average_loss:.4f}"
        )

    print("\nEvaluating model...")

    model.eval()
    predictions = []
    actual_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)

            outputs = model(inputs)
            probabilities = torch.sigmoid(outputs)
            predicted_labels = (probabilities >= 0.5).int()

            predictions.extend(predicted_labels.cpu().tolist())
            actual_labels.extend(labels.int().tolist())

    print(f"Accuracy: {accuracy_score(actual_labels, predictions):.4f}")
    print(classification_report(
        actual_labels,
        predictions,
        target_names=["Fake", "Real"]
    ))

    MODEL_DIR.mkdir(exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": len(vocab)
        },
        MODEL_DIR / "fake_news_model.pt"
    )

    joblib.dump(vocab, MODEL_DIR / "vocab.pkl")

    print("\nTraining complete.")
    print("Model saved in the models folder.")


if __name__ == "__main__":
    main()