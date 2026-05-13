import os
import pickle
from collections import Counter, defaultdict

import numpy as np
from sklearn.decomposition import TruncatedSVD

CORPUS_PATH = "data/processed/corpus_clean.txt"
MODEL_PATH = "models/glove_vectors.pkl"

VECTOR_SIZE = 100
WINDOW_SIZE = 5
MIN_COUNT = 2
MAX_VOCAB = 8000

os.makedirs("models", exist_ok=True)

sentences = []

with open(CORPUS_PATH, "r", encoding="utf-8") as f:
    for line in f:
        tokens = line.strip().split()
        if len(tokens) > 2:
            sentences.append(tokens)

print(f"Jumlah kalimat: {len(sentences)}")

word_counts = Counter()

for sentence in sentences:
    word_counts.update(sentence)

vocab_words = [
    word for word, count in word_counts.most_common(MAX_VOCAB)
    if count >= MIN_COUNT
]

word_to_id = {word: idx for idx, word in enumerate(vocab_words)}
id_to_word = {idx: word for word, idx in word_to_id.items()}

vocab_size = len(vocab_words)
print(f"Jumlah vocabulary: {vocab_size}")

cooc = defaultdict(float)

print("Membuat co-occurrence matrix...")

for sentence in sentences:
    tokens = [word for word in sentence if word in word_to_id]

    for i, word in enumerate(tokens):
        word_id = word_to_id[word]

        start = max(0, i - WINDOW_SIZE)
        end = min(len(tokens), i + WINDOW_SIZE + 1)

        for j in range(start, end):
            if i == j:
                continue

            context_word = tokens[j]
            context_id = word_to_id[context_word]
            distance = abs(i - j)

            cooc[(word_id, context_id)] += 1.0 / distance

matrix = np.zeros((vocab_size, vocab_size), dtype=np.float32)

for (i, j), value in cooc.items():
    matrix[i, j] = value

print("Membentuk embedding GloVe sederhana dengan SVD...")

svd = TruncatedSVD(n_components=VECTOR_SIZE, random_state=42)
vectors = svd.fit_transform(matrix)

model_data = {
    "vectors": vectors,
    "word_to_id": word_to_id,
    "id_to_word": id_to_word
}

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model_data, f)

print("\nTraining GloVe selesai.")
print(f"Model disimpan di: {MODEL_PATH}")

def most_similar(word, topn=5):
    if word not in word_to_id:
        return []

    word_id = word_to_id[word]
    target_vector = vectors[word_id]

    similarities = vectors @ target_vector
    norms = np.linalg.norm(vectors, axis=1) * np.linalg.norm(target_vector)
    similarities = similarities / (norms + 1e-9)

    top_ids = similarities.argsort()[::-1][1:topn + 1]

    return [(id_to_word[idx], similarities[idx]) for idx in top_ids]

print("\nContoh similar words untuk 'bagus':")

results = most_similar("bagus", topn=5)

if results:
    for word, score in results:
        print(f"{word} -> {score:.4f}")
else:
    print("Kata 'bagus' tidak ditemukan.")