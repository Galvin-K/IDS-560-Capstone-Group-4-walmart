import pathlib
import sys
import unittest
import unittest.mock as mock
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import v4_policy_bridge as bridge

class FakeV4:
    def baseline_forecast(self,item,store,date):
        return {'mean':[2.0]*14,'model':'LightGBM 14-day fixture'}
    def recommend(self,*args,**kwargs):
        assert kwargs['call_llm'] is False
        return {'order':{'computed_order_qty':25,'order_qty':0,'hold':True},
                'confidence':{'level':'Low'}, 'flags':[{'code':'REVIEW'}]}

class Tests(unittest.TestCase):
    def test_not_enabled(self):
        with mock.patch.dict('os.environ',{'V4_POLICY_ENABLED':'0'}):
            self.assertFalse(bridge.policy_status()[0])
            with self.assertRaises(RuntimeError):
                bridge.v4_recommendation('A','WI_1','2015-12-31',10)
    def test_official_adapter_preserves_hold(self):
        with mock.patch.object(bridge,'policy_status',return_value=(True,'ready')):
            with mock.patch.object(bridge,'_load_v4_policy',return_value=FakeV4()):
                result=bridge.v4_recommendation('A','WI_1','2015-12-31',10)
        self.assertEqual(result['forecast_14_days'],28.0)
        self.assertTrue(result['order']['hold'])
        self.assertEqual(result['confidence']['level'],'Low')
    def test_rejects_incomplete_forecast(self):
        class Short(FakeV4):
            def baseline_forecast(self,*args): return {'mean':[1]*7,'model':'short'}
        with mock.patch.object(bridge,'policy_status',return_value=(True,'ready')):
            with mock.patch.object(bridge,'_load_v4_policy',return_value=Short()):
                with self.assertRaisesRegex(RuntimeError,'Exact 14-day'):
                    bridge.v4_recommendation('A','WI_1','2015-12-31',10)

if __name__=='__main__':unittest.main()
