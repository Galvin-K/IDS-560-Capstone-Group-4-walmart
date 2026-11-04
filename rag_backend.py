"""Historical evidence retrieval for Walmart M5 decision support.

This remains compatible with the deployed v3 index and chunks.  The v4
notebooks use a separate rag_common.py with a reranker; their full retrieval
stack cannot be reproduced from the five notebooks alone.
"""
from pathlib import Path
from functools import lru_cache
import pickle
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

RAG_DIR = Path(__file__).resolve().parent / 'RAG'
CHUNKS_FILE = RAG_DIR / 'chunks_v3.pkl'
INDEX_FILE = RAG_DIR / 'index_v3.pkl'

MAX_AGE_DAYS = {
    'storm_event': 45, 'beige_book': 120, 'sec_filing': 200,
    'walmart_doc': 400, 'policy_doc': 400, 'evidence_card': None,
    'reference': None, 'extra': None,
}

@lru_cache(maxsize=1)
def load_rag():
    """Load trusted project-created pickles only; keep model cached per process."""
    chunks = pd.read_pickle(CHUNKS_FILE).reset_index(drop=True)
    with open(INDEX_FILE, 'rb') as f:
        index = pickle.load(f)
    chunks['published'] = pd.to_datetime(chunks['published'], errors='coerce')
    if chunks['chunk_id'].tolist() != index['chunk_ids']:
        raise ValueError('RAG chunks and index do not match.')
    if index['kind'] != 'dense':
        raise ValueError('Expected a dense embedding index.')
    vectors = np.asarray(index['emb'], dtype=np.float32)
    if vectors.shape[0] != len(chunks):
        raise ValueError('Embedding count does not match passages.')
    model = SentenceTransformer(index['model'])
    return chunks, vectors, model


def retrieve(query, as_of, state=None, k=5, category=None,
             min_similarity=None, category_filter=True):
    """Return as-of-safe, diverse semantic results, optionally category-aware.

    min_similarity applies to the v3 cosine/dot-product similarity, NOT the
    v4 reranker's calibrated relevance measure. Set to None by default.
    """
    chunks, embeddings, model = load_rag()
    cutoff = pd.Timestamp(as_of)
    eligible = chunks['published'].notna() & (chunks['published'] <= cutoff)
    if state:
        eligible &= chunks['state'].isin([state, 'US'])
    if category_filter and category and category.upper() in ('FOODS','HOBBIES','HOUSEHOLD') and 'category' in chunks:
        eligible &= chunks['category'].fillna('ALL').isin(['ALL', category.upper()])

    ages = (cutoff - chunks['published']).dt.days
    for doc_type, max_days in MAX_AGE_DAYS.items():
        if max_days is not None:
            eligible &= ~((chunks['doc_type'] == doc_type) & (ages > max_days))

    candidates = chunks.loc[eligible].copy()
    if 'series_key' in candidates:
        keyed = candidates['series_key'].notna()
        latest = candidates.loc[keyed].groupby('series_key')['published'].transform('max')
        stale = latest.index[candidates.loc[keyed, 'published'] < latest]
        candidates = candidates.drop(index=stale)
    if candidates.empty:
        return candidates.assign(score=pd.Series(dtype=float))

    query_text = 'Represent this sentence for searching relevant passages: ' + str(query)
    query_vector = model.encode([query_text], normalize_embeddings=True)[0].astype(np.float32)
    candidates['score'] = embeddings[candidates.index.to_numpy()] @ query_vector
    if min_similarity is not None:
        candidates = candidates.loc[candidates['score'] >= float(min_similarity)]
    candidates = candidates.sort_values('score', ascending=False)

    selected, doc_counts, type_counts = [], {}, {}
    for idx, row in candidates.iterrows():
        doc_id, doc_type = row['doc_id'], row['doc_type']
        if doc_counts.get(doc_id, 0) >= 1 or type_counts.get(doc_type, 0) >= 3:
            continue
        selected.append(idx)
        doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1
        type_counts[doc_type] = type_counts.get(doc_type, 0) + 1
        if len(selected) >= max(1, int(k)):
            break
    return candidates.loc[selected].copy()
