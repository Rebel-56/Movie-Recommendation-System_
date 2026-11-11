"""Streamlit web app for the movie recommender.

Run:  streamlit run app.py
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).parent / "src"))
from recommender import MovieRecommender  # noqa: E402

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="centered")


@st.cache_resource
def load_model() -> MovieRecommender:
    return MovieRecommender()


rec = load_model()

st.title("🎬 Movie Recommender System")
st.caption("Content-based filtering using TF-IDF and cosine similarity")

tab1, tab2 = st.tabs(["Similar movies", "Describe what you want"])

with tab1:
    movie = st.selectbox("Pick a movie you like", rec.titles())
    n = st.slider("Number of recommendations", 3, 10, 5, key="n1")
    if st.button("Recommend", type="primary"):
        st.dataframe(rec.recommend(movie, n), use_container_width=True, hide_index=True)

with tab2:
    query = st.text_input("e.g. 'friends road trip comedy' or 'detective hunts a serial killer'")
    n2 = st.slider("Number of results", 3, 10, 5, key="n2")
    if query:
        st.dataframe(rec.search(query, n2), use_container_width=True, hide_index=True)
