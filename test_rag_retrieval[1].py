"""Minimal offline tests for the modified RAG retrieval selection logic."""
import sys
from types import ModuleType
import numpy as np
import pandas as pd

stub = ModuleType('sentence_transformers')
stub.SentenceTransformer = object
sys.modules['sentence_transformers'] = stub
import rag_backend as rb

class Model:
    def encode(self, queries, normalize_embeddings=True):
        return np.array([[1., 0.]], dtype=np.float32)

chunks = pd.DataFrame([
    dict(chunk_id='a',published='2015-12-30',state='CA',category='FOODS',doc_type='evidence_card',doc_id='1',series_key=None,score=0),
    dict(chunk_id='b',published='2015-12-31',state='US',category='ALL',doc_type='reference',doc_id='2',series_key=None,score=0),
    dict(chunk_id='c',published='2016-01-02',state='CA',category='FOODS',doc_type='evidence_card',doc_id='3',series_key=None,score=0),
    dict(chunk_id='d',published='2015-12-30',state='TX',category='FOODS',doc_type='evidence_card',doc_id='4',series_key=None,score=0),
    dict(chunk_id='e',published='2015-12-30',state='CA',category='HOBBIES',doc_type='evidence_card',doc_id='5',series_key=None,score=0),
])
chunks['published'] = pd.to_datetime(chunks['published'])
vectors = np.array([[.9,.1],[.7,.3],[1.,0.],[.8,.2],[.95,.05]], dtype=np.float32)
rb.load_rag = lambda: (chunks, vectors, Model())
result=rb.retrieve('Foods California','2015-12-31',state='CA',category='FOODS',k=10)
assert result['chunk_id'].tolist()==['a','b'], result['chunk_id'].tolist()
assert result['published'].le(pd.Timestamp('2015-12-31')).all()
assert rb.retrieve('any','2015-12-31',state='CA',category='FOODS',min_similarity=.8)['chunk_id'].tolist()==['a']
assert rb.retrieve('any','2015-12-31',state='CA',category='FOODS',k=1)['chunk_id'].tolist()==['a']
assert rb.retrieve('any','2015-12-31',state='CA',category=None,k=8)['chunk_id'].tolist()==['e','a','b']
print('RAG retrieval offline tests passed')
