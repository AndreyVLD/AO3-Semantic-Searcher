import streamlit as st
import pandas as pd

from numpy import ndarray
from src.db import WorkRepository
from src.config import WORKS_DB, TOP_K
from src.embedding import EmbeddingModel


def get_db() -> WorkRepository:
    """
    Get a database connection to the works repository.
    """
    return WorkRepository(WORKS_DB)


@st.cache_resource
def get_embedding_model() -> EmbeddingModel:
    """
    Get an instance of the embedding model.
    """
    return EmbeddingModel()


st.set_page_config(
    page_title="AO3 Search",
    page_icon=":open_book:",
    layout="wide",
)
st.title("AO3 Semantic Search")

with st.form("search_form"):
    col1, col2 = st.columns([0.94, 0.06], vertical_alignment="bottom")
    with col1:
        query = st.text_input("Search query", help="Type keywords to search")
    with col2:
        submitted = st.form_submit_button("Submit")

db = get_db()
model = get_embedding_model()

if submitted and query:
    status = st.empty()
    bar = st.progress(0)

    with st.spinner("Embedding user input...", show_time=True):
        query_embedding: ndarray = model.embed_chunks([query])[0]

    status.success("User input embedded successfully.")
    bar.progress(25)

    with st.spinner(f"Retrieving top {TOP_K} works...", show_time=True):
        retrieved_works = db.retrieve_top_k_works(query_embedding, top_k=TOP_K)

    status.success(f"Retrieved {len(retrieved_works)} relevant works.")
    bar.progress(50)

    with st.spinner("Re-Ranking works...", show_time=True):
        pairs = [(query, str(work)) for work in retrieved_works]
        scores = model.cross_scores(pairs)

        for work, score in zip(retrieved_works, scores):
            work.score = score

        retrieved_works.sort(key=lambda x: x.score, reverse=True)

    status.success("Re-Ranking completed.")
    bar.progress(75)

    with st.spinner("Displaying results...", show_time=True):
        display_works = []
        for work in retrieved_works:
            display_works.append({
                "Score": f"{work.score:.4f}",
                "Title": work.title,
                "Author": work.author,
                "Category": work.category,
                "Genre": work.genre,
                "Relationships": work.relationships or "",
                "Summary": work.summary or "",
                "Rating": work.rating,
                "Warnings": work.warnings,
                "Series": work.series or "",
                "Collections": work.collections or "",
                "Story URL": work.storyURL or "",
            })
        st.dataframe(pd.DataFrame(display_works), use_container_width=True, height=((len(display_works) + 1) * 35 + 3))

    status.success("Final results are complete.")
    bar.progress(100)

    bar.empty()
