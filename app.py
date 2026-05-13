import pickle

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from gensim.models import FastText
from sklearn.decomposition import PCA

FASTTEXT_PATH = "models/fasttext_model.model"
GLOVE_PATH = "models/glove_vectors.pkl"

st.set_page_config(
    page_title="GloVe vs FastText",
    layout="wide"
)

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
.title {
    font-size: 38px;
    font-weight: 800;
    color: #111827;
    margin-bottom: 4px;
}
.subtitle {
    font-size: 16px;
    color: #6b7280;
    margin-bottom: 26px;
}
.panel {
    background-color: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 18px;
}
.section-title {
    font-size: 22px;
    font-weight: 700;
    color: #111827;
    margin-bottom: 12px;
}
.small-text {
    color: #6b7280;
    font-size: 14px;
}
.stButton button {
    width: 100%;
    height: 44px;
    border-radius: 10px;
    font-weight: 600;
}
[data-testid="stMetricValue"] {
    font-size: 24px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_fasttext():
    return FastText.load(FASTTEXT_PATH)


@st.cache_resource
def load_glove():
    with open(GLOVE_PATH, "rb") as f:
        return pickle.load(f)


def cosine_similarity(vectors, target_vector):
    similarities = vectors @ target_vector
    norms = np.linalg.norm(vectors, axis=1) * np.linalg.norm(target_vector)
    return similarities / (norms + 1e-9)


def glove_most_similar(glove_data, word, topn=10):
    vectors = glove_data["vectors"]
    word_to_id = glove_data["word_to_id"]
    id_to_word = glove_data["id_to_word"]

    if word not in word_to_id:
        return []

    word_id = word_to_id[word]
    target_vector = vectors[word_id]

    similarities = cosine_similarity(vectors, target_vector)
    top_ids = similarities.argsort()[::-1][1:topn + 1]

    return [(id_to_word[idx], float(similarities[idx])) for idx in top_ids]


def fasttext_most_similar(model, word, topn=10):
    try:
        return [(word, float(score)) for word, score in model.wv.most_similar(word, topn=topn)]
    except KeyError:
        return []


def get_fasttext_vectors(model, words):
    vectors = []
    valid_words = []

    for word in words:
        try:
            vectors.append(model.wv[word])
            valid_words.append(word)
        except KeyError:
            pass

    return valid_words, np.array(vectors)


def get_glove_vectors(glove_data, words):
    vectors_data = glove_data["vectors"]
    word_to_id = glove_data["word_to_id"]

    vectors = []
    valid_words = []

    for word in words:
        if word in word_to_id:
            vectors.append(vectors_data[word_to_id[word]])
            valid_words.append(word)

    return valid_words, np.array(vectors)


def plot_interactive_pca(words, vectors, title, input_word):
    if len(words) < 2:
        st.warning("Jumlah kata terlalu sedikit untuk divisualisasikan.")
        return

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(vectors)

    df_plot = pd.DataFrame({
        "Kata": words,
        "PCA 1": reduced[:, 0],
        "PCA 2": reduced[:, 1],
        "Jenis": ["Input"] + ["Similar Word"] * (len(words) - 1)
    })

    fig = px.scatter(
        df_plot,
        x="PCA 1",
        y="PCA 2",
        text="Kata",
        hover_name="Kata",
        color="Jenis",
        title=title,
        height=520
    )

    fig.update_traces(
        textposition="top center",
        marker=dict(size=13, line=dict(width=1))
    )

    fig.update_layout(
        title=dict(
            x=0.02,
            font=dict(size=20)
        ),
        legend_title_text="Kategori",
        margin=dict(l=20, r=20, t=60, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(showgrid=True, zeroline=False),
        yaxis=dict(showgrid=True, zeroline=False)
    )

    st.plotly_chart(fig, use_container_width=True)


def show_model_result(title, results, input_word, model_type, fasttext_model=None, glove_data=None):
    st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)

    if not results:
        st.warning(f"Kata '{input_word}' tidak ditemukan pada model {model_type}.")
        return

    df = pd.DataFrame(results, columns=["Kata", "Similarity"])
    df["Similarity"] = df["Similarity"].round(4)

    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric("Jumlah kata", len(df))

    with m2:
        st.metric("Kata terdekat", df.iloc[0]["Kata"])

    with m3:
        st.metric("Skor tertinggi", df.iloc[0]["Similarity"])

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    words = [input_word] + df["Kata"].tolist()

    if model_type == "FastText":
        valid_words, vectors = get_fasttext_vectors(fasttext_model, words)
    else:
        valid_words, vectors = get_glove_vectors(glove_data, words)

    plot_interactive_pca(
        valid_words,
        vectors,
        f"Visualisasi PCA Interaktif - {model_type}",
        input_word
    )


try:
    fasttext_model = load_fasttext()
    glove_data = load_glove()
except Exception as e:
    st.error("Model belum ditemukan. Jalankan preprocessing dan training terlebih dahulu.")
    st.code(str(e))
    st.stop()


st.markdown("<div class='title'>GloVe vs FastText</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Interface perbandingan word embedding berdasarkan dataset review Bahasa Indonesia.</div>",
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("Pengaturan")

    model_choice = st.selectbox(
        "Pilih model",
        ["Bandingkan Keduanya", "FastText", "GloVe"]
    )

    input_word = st.text_input("Masukkan kata", value="bagus")
    topn = st.slider("Jumlah kata terdekat", 5, 20, 10)

    search_button = st.button("Cari Similar Words")


st.markdown("<div class='panel'>", unsafe_allow_html=True)

st.markdown("### Ringkasan")
col_a, col_b, col_c = st.columns(3)

with col_a:
    st.metric("Model 1", "FastText")
    st.caption("Menggunakan subword sehingga lebih kuat untuk variasi kata.")

with col_b:
    st.metric("Model 2", "GloVe")
    st.caption("Menggunakan hubungan kemunculan kata dalam corpus.")

with col_c:
    st.metric("Visualisasi", "PCA 2D")
    st.caption("Grafik interaktif untuk melihat kedekatan antar kata.")

st.markdown("</div>", unsafe_allow_html=True)


if search_button:
    input_word = input_word.lower().strip()

    st.markdown(f"### Hasil pencarian: `{input_word}`")

    if model_choice == "FastText":
        results = fasttext_most_similar(fasttext_model, input_word, topn)

        show_model_result(
            "FastText",
            results,
            input_word,
            "FastText",
            fasttext_model=fasttext_model
        )

    elif model_choice == "GloVe":
        results = glove_most_similar(glove_data, input_word, topn)

        show_model_result(
            "GloVe",
            results,
            input_word,
            "GloVe",
            glove_data=glove_data
        )

    else:
        fasttext_results = fasttext_most_similar(fasttext_model, input_word, topn)
        glove_results = glove_most_similar(glove_data, input_word, topn)

        col1, col2 = st.columns(2)

        with col1:
            show_model_result(
                "FastText",
                fasttext_results,
                input_word,
                "FastText",
                fasttext_model=fasttext_model
            )

        with col2:
            show_model_result(
                "GloVe",
                glove_results,
                input_word,
                "GloVe",
                glove_data=glove_data
            )

else:
    st.info("Masukkan kata di sidebar, pilih model, lalu klik tombol Cari Similar Words.")