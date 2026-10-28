# Walmart M5 - RAG Evidence Retrieval

from pathlib import Path
from functools import lru_cache
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

RAG_DIR = (Path(__file__).resolve().parent / "RAG")

CHUNKS_FILE = RAG_DIR / "chunks_v3.pkl"
INDEX_FILE = RAG_DIR / "index_v3.pkl"

MAX_AGE_DAYS = {
    "storm_event": 45,
    "beige_book": 120,
    "sec_filing": 200,
    "walmart_doc": 400,
    "policy_doc": 400,
    "evidence_card": None,
    "reference": None,
    "extra": None
}


@lru_cache(maxsize=1)
def load_rag():
    # Load only trusted project-generated pickle files
    chunks = pd.read_pickle(CHUNKS_FILE).reset_index(drop=True)

    with open(INDEX_FILE, "rb") as f:
        index = pickle.load(f)

    chunks["published"] = pd.to_datetime(chunks["published"])

    if chunks["chunk_id"].tolist() != index["chunk_ids"]:
        raise ValueError("RAG chunks and index do not match.")

    if index["kind"] != "dense":
        raise ValueError("Expected a dense embedding index.")

    embeddings = np.asarray(index["emb"], dtype=np.float32)

    if embeddings.shape[0] != len(chunks):
        raise ValueError("Embedding count does not match passages.")

    model = SentenceTransformer(index["model"])

    return chunks, embeddings, model


def retrieve(query, as_of, state=None, k=6):
    chunks, embeddings, model = load_rag()

    as_of = pd.Timestamp(as_of)

    # Exclude documents published after the decision date
    eligible = chunks["published"] <= as_of

    # Keep documents relevant to the state or US-wide
    if state:
        eligible &= chunks["state"].isin([state, "US"])

    # Apply Tyler's document-age restrictions
    age = (as_of - chunks["published"]).dt.days

    for doc_type, max_days in MAX_AGE_DAYS.items():
        if max_days is not None:
            eligible &= ~(
                (chunks["doc_type"] == doc_type)
                & (age > max_days)
            )

    candidates = chunks.loc[eligible].copy()

    # Keep the latest available version of each evidence card
    if "series_key" in candidates.columns:
        keyed = candidates["series_key"].notna()

        latest = candidates.loc[keyed].groupby(
            "series_key"
        )["published"].transform("max")

        stale = latest.index[
            candidates.loc[keyed, "published"] < latest
        ]

        candidates = candidates.drop(index=stale)

    if candidates.empty:
        return candidates.assign(score=pd.Series(dtype=float))

    # Match Tyler's BGE query encoding
    query_text = (
        "Represent this sentence for searching relevant passages: "
        + query
    )

    query_vector = model.encode(
        [query_text],
        normalize_embeddings=True
    )[0].astype(np.float32)

    # Rank the eligible documents by similarity
    candidate_vectors = embeddings[
        candidates.index.to_numpy()
    ]

    candidates["score"] = candidate_vectors @ query_vector

    candidates = candidates.sort_values(
        "score", ascending=False
    )

    # Limit repeated documents and evidence types
    selected = []
    doc_counts = {}
    type_counts = {}

    for idx, row in candidates.iterrows():
        doc_id = row["doc_id"]
        doc_type = row["doc_type"]

        if doc_counts.get(doc_id, 0) >= 1:
            continue

        if type_counts.get(doc_type, 0) >= 3:
            continue

        selected.append(idx)
        doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1
        type_counts[doc_type] = type_counts.get(doc_type, 0) + 1

        if len(selected) >= k:
            break

    return candidates.loc[selected].copy()
