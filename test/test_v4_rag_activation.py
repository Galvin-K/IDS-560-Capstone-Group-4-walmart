"""Lightweight tests: mock the expensive transformers instead of loading them."""
import importlib.util
import os
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('v4_activation_backend', ROOT / 'rag_backend.py')
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)


class FakeCommon:
    RELEVANCE_MIN = .0153
    RERANKER = object()
    def __init__(self):
        self.kwargs = None
    def retrieve(self, query, as_of, state=None, k=5, category=None, method='full'):
        self.kwargs = {'k':k,'category':category,'method':method}
        return pd.DataFrame([
            {'title':'Good', 'published':'2015-11-15', 'relevance': .045, 'score':.023},
            {'title':'Unrelated', 'published':'2015-11-01', 'relevance':.001, 'score':.04},
            {'title':'Future', 'published':'2016-02-01', 'relevance':.99, 'score':.99},
        ])

class ActivationTests(unittest.TestCase):
    def test_disabled_by_default(self):
        with patch.object(backend, '_v4_requested', return_value=False):
            self.assertFalse(backend.retrieval_status()['v4'])

    def test_v4_requires_actual_reranker(self):
        fake = types.SimpleNamespace(setup=lambda **kw:None, retrieve=lambda *a,**k:None, RERANKER=None)
        backend._v4.cache_clear()
        with patch.object(backend, '_v4_requested', return_value=True), \
             patch.object(backend.importlib, 'import_module', return_value=fake), \
             patch.object(backend.Path, 'is_file', return_value=True):
            self.assertFalse(backend.retrieval_status()['v4'])
        backend._v4.cache_clear()

    def test_date_relevance_and_candidate_limit(self):
        fake = FakeCommon()
        with patch.object(backend, 'retrieval_status', return_value={'v4':True}), \
             patch.object(backend, '_v4', return_value=fake):
            df = backend.retrieve('foods', '2015-12-31', state='CA', category='FOODS', k=5)
        self.assertEqual(df.title.tolist(), ['Good'])
        self.assertEqual(float(df.iloc[0].score), .045)
        self.assertEqual(fake.kwargs, {'k':30,'category':'FOODS','method':'full'})

    def test_provisional_cutoff_could_return_no_passages(self):
        fake = FakeCommon()
        fake.RELEVANCE_MIN = .1
        with patch.object(backend, 'retrieval_status', return_value={'v4':True}), \
             patch.object(backend, '_v4', return_value=fake):
            df = backend.retrieve('foods', '2015-12-31', k=3)
        self.assertTrue(df.empty)

    def test_streamlit_secret_precedence(self):
        fake_st = types.SimpleNamespace(secrets={'RAG_USE_V4':'1'})
        with patch.dict(sys.modules, {'streamlit':fake_st}), patch.dict(os.environ, {'RAG_USE_V4':'0'}):
            self.assertTrue(backend._v4_requested())

if __name__ == '__main__':
    unittest.main()
