from gensim.models import FastText
import os

CORPUS_PATH = "data/processed/corpus_clean.txt"
MODEL_PATH = "models/fasttext_model.model"

os.makedirs("models", exist_ok=True)

sentences = []

with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    for line in f:
        tokens = line.strip().split()
        if len(tokens) > 2:
            sentences.append(tokens)

print(f"Jumlah kalimat: {len(sentences)}")
print("Training FastText...")

model = FastText(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    sg=1,
    epochs=20
)

model.save(MODEL_PATH)

print("\nTraining FastText selesai.")
print(f"Model disimpan di: {MODEL_PATH}")

try:
    print("\nContoh similar words untuk 'bagus':")
    for word, score in model.wv.most_similar("bagus", topn=5):
        print(f"{word} -> {score:.4f}")
except KeyError:
    print("Kata 'bagus' tidak ditemukan.")