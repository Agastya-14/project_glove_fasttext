# GloVe vs FastText - Indonesian NLP Project

Project ini merupakan implementasi sederhana word embedding menggunakan:

- FastText
- GloVe

dengan dataset review Bahasa Indonesia dan interface interaktif menggunakan Streamlit.

---

## Dataset

Dataset yang digunakan:

[PRDECT-ID Indonesian Emotion Classification Dataset](https://www.kaggle.com/datasets/jocelyndumlao/prdect-id-indonesian-emotion-classification?utm_source=chatgpt.com)

Dataset berisi review Bahasa Indonesia yang digunakan untuk:
- preprocessing text,
- training word embedding,
- similarity analysis,
- visualisasi embedding.

---

## Features

- Training FastText model
- Training GloVe model sederhana
- Similar word search
- Perbandingan FastText vs GloVe
- Interactive PCA Visualization
- Streamlit Dashboard Interface

---

## Project Structure

```text
project_glove_fasttext/
│
├── data/
│   ├── raw/
│   │   └── dataset.csv
│   │
│   └── processed/
│       └── corpus_clean.txt
│
├── models/
│   ├── fasttext_model.model
│   └── glove_vectors.pkl
│
├── preprocessing.py
├── train_fasttext.py
├── train_glove.py
├── app.py
├── requirements.txt
└── README.md


Installation
1. Create Virtual Environment
py -3.11 -m venv venv

2. Activate Virtual Environment
venv\Scripts\activate

3. Install Dependencies
pip install -r requirements.txt


Run Project
1. Preprocessing Dataset
python preprocessing.py

2. Train FastText
python train_fasttext.py

3. Train GloVe
python train_glove.py

4. Run Streamlit Interface
streamlit run app.py


Technologies Used
- Python 3.11
- Streamlit
- Gensim
- Scikit-learn
- Plotly
- Pandas
- NumPy


Visualization
Project menggunakan:
- PCA (Principal Component Analysis)
- Interactive Plotly Visualization
untuk memvisualisasikan hubungan antar kata pada embedding space.


Notes
- Dataset tidak disertakan pada repository GitHub.
- Download dataset terlebih dahulu dari Kaggle.
Simpan dataset pada:
data/raw/dataset.csv


Author
Efza Nur Agastya
