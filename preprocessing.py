import pandas as pd
import re
import os

RAW_PATH = "data/raw/dataset.csv"
OUTPUT_PATH = "data/processed/corpus_clean.txt"
TEXT_COLUMN = "Customer Review"

os.makedirs("data/processed", exist_ok=True)

df = pd.read_csv(RAW_PATH)

print("Kolom dataset:")
print(df.columns)

if TEXT_COLUMN not in df.columns:
    print(f"\nKolom '{TEXT_COLUMN}' tidak ditemukan.")
    print("Cek kembali nama kolom dataset.")
    exit()

df = df[[TEXT_COLUMN]].dropna()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_text"] = df[TEXT_COLUMN].apply(clean_text)
df = df[df["clean_text"].str.split().str.len() > 2]

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for text in df["clean_text"]:
        f.write(text + "\n")

print("\nPreprocessing selesai.")
print(f"Jumlah data: {len(df)}")
print(f"Corpus disimpan di: {OUTPUT_PATH}")