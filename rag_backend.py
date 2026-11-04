"""Date-safe RAG retrieval with an optional exact v4 implementation.

When the team's `rag_common.py` and its artifacts are deployed, full v4
retrieval can be selected with RAG_USE_V4=1. Otherwise use a working,
independently implemented v3-index HYBRID candidate ranker (dense + lexical).
The fallback is NOT the team's evaluated v4 reranker and has no calibrated
no-answer threshold.

Only load pickles from the trusted, maintained project repository.
"""
from functools import lru_cache
from pathlib import Path
from collections import Counter
import importlib
import os
import pickle
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RAG_DIR = ROOT / 'RAG'
V3_CHUNKS = RAG_DIR / 'chunks_v3.pkl'
V3_INDEX = RAG_DIR / 'index_v3.pkl'
MAX_AGE_DAYS = {
    'storm_event': 45, 'beige_book': 120, 'sec_filing': 200,
    'walmart_doc': 400, 'policy_doc': 400, 'evidence_card': None,
    'reference': None, 'extra': None,
}
_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = {'what', 'when', 'why', 'which', 'from', 'with', 'this', 'that', 'there',
         'were', 'have', 'would', 'could', 'should', 'about', 'during', 'might',
         'walmart', 'sales', 'demand', 'historical', 'product', 'products'}


def _tokens(s):
    return {t for t in _TOKEN.findall(str(s).lower()) if len(t) > 2 and t not in _STOP}


@lru_cache(maxsize=1)
def _v4():
    """Load official v4 only when operator opts in and dependencies exist."""
    if os.getenv('RAG_USE_V4', '0') != '1':
        raise RuntimeError('v4 not enabled; set RAG_USE_V4=1 after deployment and testing')
    if not (RAG_DIR / 'rag_common.py').is_file() and not (ROOT / 'rag_common.py').is_file():
        raise FileNotFoundError('rag_common.py is missing')
    import sys
    for directory in (str(RAG_DIR), str(ROOT)):
        if directory not in sys.path:
            sys.path.insert(0, directory)
    rc = importlib.import_module('rag_common')
    # The supplied v4 notebooks use this setup contract.
    rc.setup(m5=False, forecasts=False, library=True)
    if not callable(getattr(rc, 'retrieve', None)):
        raise RuntimeError('rag_common.retrieve unavailable')
    return rc


def retrieval_status():
    """Truthful active mode; missing/invalid v4 never silently passes as deployed."""
    if os.getenv('RAG_USE_V4', '0') != '1':
        return {'mode': 'hybrid-v3', 'v4': False,
                'message': 'Dense + keyword ranking on v3 evidence; official v4 reranker is not active.'}
    try:
        _v4()
        return {'mode': 'official-v4', 'v4': True,
                'message': 'Official v4 retrieval module initialized; verify evaluations and configuration.'}
    except Exception as exc:
        return {'mode': 'hybrid-v3', 'v4': False,
                'message': f'Official v4 unavailable ({type(exc).__name__}: {exc}); using hybrid v3 fallback.'}


@lru_cache(maxsize=1)
def load_rag():
    """Load the current trusted v3 evidence; cache one model per process."""
    from sentence_transformers import SentenceTransformer
    chunks = pd.read_pickle(V3_CHUNKS).reset_index(drop=True)
    with open(V3_INDEX, 'rb') as f:
        index = pickle.load(f)
    if chunks['chunk_id'].tolist() != index['chunk_ids']:
        raise ValueError('Chunks and index do not match')
    if index['kind'] != 'dense':
        raise ValueError('Dense index expected')
    embeddings = np.asarray(index['emb'], np.float32)
    if len(embeddings) != len(chunks):
        raise ValueError('Embedding count mismatch')
    chunks['published'] = pd.to_datetime(chunks['published'], errors='coerce')
    return chunks, embeddings, SentenceTransformer(index['model'])


def _eligible(chunks, cutoff, state=None, category=None, category_filter=True):
    cutoff = pd.Timestamp(cutoff)
    valid = chunks['published'].notna() & (chunks['published'] <= cutoff)
    if state:
        valid &= chunks['state'].isin([state, 'US'])
    if category_filter and category and 'category' in chunks:
        cat = str(category).upper()
        if cat in ('FOODS', 'HOBBIES', 'HOUSEHOLD'):
            valid &= chunks['category'].fillna('ALL').isin(['ALL', cat])
    age = (cutoff - chunks['published']).dt.days
    for kind, max_age in MAX_AGE_DAYS.items():
        if max_age is not None:
            valid &= ~((chunks['doc_type'] == kind) & (age > max_age))
    candidates = chunks.loc[valid].copy()
    if 'series_key' in candidates:
        keyed = candidates['series_key'].notna()
        recent = candidates.loc[keyed].groupby('series_key')['published'].transform('max')
        stale = recent.index[candidates.loc[keyed, 'published'] < recent]
        candidates = candidates.drop(index=stale)
    return candidates


def _rank_hybrid(candidates, query, vector_scores):
    """Reciprocal-rank fusion of cosine dense scores and lexical overlap.

    Unlike v4's evaluated cross-encoder, this is an *unvalidated* transparent
    fallback. Scores are ranking-only and cannot be thresholded as relevance.
    """
    q = _tokens(query)
    lexical = []
    for _, row in candidates.iterrows():
        title = _tokens(row.get('title', ''))
        body = _tokens(str(row.get('text', ''))[:4000])
        matched = len(q & body) + 2 * len(q & title)
        lexical.append(matched / max(1, len(q)))
    lexical = np.asarray(lexical, dtype=float)
    dense = np.asarray(vector_scores, dtype=float)
    dr = pd.Series(dense).rank(ascending=False, method='min').to_numpy()
    lr = pd.Series(lexical).rank(ascending=False, method='min').to_numpy()
    # Suppress a lexical-only ranking signal when no query words occur.
    fused = 1.0 / (60.0 + dr)
    if np.any(lexical > 0):
        fused += 1.0 / (60.0 + lr)
    return fused, lexical


def _retrieve_fallback(query, as_of, state, k, category, min_similarity, category_filter, method):
    chunks, embeddings, model = load_rag()
    candidates = _eligible(chunks, as_of, state, category, category_filter)
    if candidates.empty:
        return candidates.assign(score=pd.Series(dtype=float), retrieval_method='hybrid-v3')
    vector = model.encode(
        ['Represent this sentence for searching relevant passages: ' + str(query)],
        normalize_embeddings=True,
    )[0].astype(np.float32)
    dense_scores = embeddings[candidates.index.to_numpy()] @ vector
    candidates['dense_similarity'] = dense_scores
    if min_similarity is not None:
        candidates = candidates.loc[candidates['dense_similarity'] >= float(min_similarity)].copy()
        dense_scores = candidates['dense_similarity'].to_numpy()
    if candidates.empty:
        return candidates.assign(score=pd.Series(dtype=float), retrieval_method='hybrid-v3')
    if method in ('hybrid', 'full'):
        combined, lexical = _rank_hybrid(candidates, query, dense_scores)
        candidates['keyword_overlap'] = lexical
        candidates['score'] = combined
        candidates['retrieval_method'] = 'hybrid-v3'
    else:
        candidates['score'] = dense_scores
        candidates['retrieval_method'] = 'dense-v3'
    candidates = candidates.sort_values('score', ascending=False)
    chosen, document_counts, type_counts = [], Counter(), Counter()
    for idx, row in candidates.iterrows():
        doc_id, doc_type = row.get('doc_id', idx), row.get('doc_type', 'unknown')
        if document_counts[doc_id] or type_counts[doc_type] >= 3:
            continue
        chosen.append(idx)
        document_counts[doc_id] += 1
        type_counts[doc_type] += 1
        if len(chosen) >= max(1, int(k)):
            break
    return candidates.loc[chosen].copy()


def retrieve(query, as_of, state=None, k=5, category=None,
             min_similarity=None, category_filter=True, method='full'):
    """Use official v4 when available; otherwise distinct labeled v3 hybrid.

    `method='full'` requests official v4 full search only if the team's shared
    implementation is configured. With missing dependencies, uses transparent
    hybrid-v3 fallback. The fallback has no reranker relevance scores.
    """
    status = retrieval_status()
    if status['v4']:
        rc = _v4()
        # Method choices are documented by the team's retrieval notebook.
        v4_method = method if method in ('dense','dense+category','hybrid+category','full') else 'full'
        res = rc.retrieve(str(query), as_of, state, k=int(k),
                          category=category if category_filter else None, method=v4_method)
        if not isinstance(res, pd.DataFrame):
            raise TypeError('v4 retrieve() must return a DataFrame')
        if not res.empty and 'published' in res:
            dates = pd.to_datetime(res['published'], errors='coerce')
            res = res.loc[dates.notna() & (dates <= pd.Timestamp(as_of))].copy()
        res = res.copy()
        res['retrieval_method'] = 'official-v4'
        if 'relevance' in res.columns:
            res['score'] = pd.to_numeric(res['relevance'], errors='coerce')
        elif 'score' not in res:
            res['score'] = np.nan
        return res
    return _retrieve_fallback(query, as_of, state, k, category,
                              min_similarity, category_filter, method)
