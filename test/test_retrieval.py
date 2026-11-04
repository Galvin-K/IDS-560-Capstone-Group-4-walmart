"""Offline integration tests: mock model, no network or API keys required."""
import pathlib
import sys
import unittest
import unittest.mock as mock
import pandas as pd
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import rag_backend as rag

class Model:
    def encode(self, queries, normalize_embeddings=True):
        return np.array([[1., 0.]], dtype=np.float32)

class Tests(unittest.TestCase):
    def setUp(self):
        rag._v4.cache_clear()
        self.rows = pd.DataFrame([
            {'chunk_id':'a', 'published':'2015-12-01', 'doc_id':'1', 'doc_type':'storm_event', 'state':'WI', 'category':'ALL', 'title':'Wisconsin winter storm weather', 'text':'Snow and winter weather in Wisconsin', 'series_key':None},
            {'chunk_id':'b', 'published':'2016-02-01', 'doc_id':'2', 'doc_type':'storm_event', 'state':'WI', 'category':'ALL', 'title':'Future Wisconsin storm', 'text':'Snow next year', 'series_key':None},
            {'chunk_id':'c', 'published':'2015-12-01', 'doc_id':'3', 'doc_type':'policy_doc', 'state':'US', 'category':'ALL', 'title':'Holiday policy weather', 'text':'Holiday season and weather', 'series_key':None},
            {'chunk_id':'d', 'published':'2015-12-15', 'doc_id':'4', 'doc_type':'evidence_card', 'state':'CA', 'category':'FOODS', 'title':'California Foods', 'text':'Food category California', 'series_key':None},
        ])
        self.rows['published'] = pd.to_datetime(self.rows['published'])
        self.embeddings = np.asarray([[.9,.1],[.99,.01],[.6,.4],[.98,.02]],dtype=np.float32)

    def test_asof_state_category_hybrid(self):
        with mock.patch.object(rag,'load_rag',return_value=(self.rows,self.embeddings,Model())):
            result = rag.retrieve('winter storm Wisconsin', '2015-12-31', state='WI', category='HOBBIES', method='full')
        self.assertEqual(set(result.chunk_id), {'a','c'})
        self.assertTrue((result['published'] <= pd.Timestamp('2015-12-31')).all())
        self.assertEqual(set(result.retrieval_method),{'hybrid-v3'})
        self.assertNotIn('relevance',result.columns)

    def test_v4_not_falsely_active_without_shared_module(self):
        with mock.patch.dict('os.environ', {'RAG_USE_V4':'0'}):
            self.assertFalse(rag.retrieval_status()['v4'])

    def test_empty_or_future_not_promoted(self):
        with mock.patch.object(rag,'load_rag',return_value=(self.rows,self.embeddings,Model())):
            result = rag.retrieve('snow','2014-01-01', state='WI')
        self.assertTrue(result.empty)

    def test_hybrid_requires_no_fixed_top_k(self):
        with mock.patch.object(rag,'load_rag',return_value=(self.rows,self.embeddings,Model())):
            result = rag.retrieve('winter storm','2015-12-31', state='WI',k=10)
        self.assertLess(len(result),10)

if __name__ == '__main__':
    unittest.main()
