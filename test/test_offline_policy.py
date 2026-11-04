import pathlib
import sys
import tempfile
import unittest
from unittest import mock
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import inventory_engine as engine
import v4_policy_bridge as bridge


def row(**updates):
    data = {
        'as_of':'2015-12-31','item_id':'FOODS_1_001','store_id':'CA_1',
        'series_id':'FOODS_1_001_CA_1_evaluation',
        'demand_14d':13.4,'safety_stock':4.0,'order_up_to':17.4,
        'new_item_if_confirmed_target':'','confidence':'Low',
        'confidence_reasons':'wide forecast','hold':False,'hold_reasons':'',
        'flags':'','model':'LightGBM 14 days','model_days':14,
        'safety_stock_method':'historical errors'
    }
    data.update(updates)
    return data

class OfflinePolicyTests(unittest.TestCase):
    def test_override_reduces_qty_and_respects_on_order(self):
        self.assertEqual(engine.apply_precomputed_policy(row(), 10)['order_qty'], 8)
        self.assertEqual(engine.apply_precomputed_policy(row(), 10, 5)['order_qty'], 3)
        self.assertEqual(engine.apply_precomputed_policy(row(), 25)['order_qty'], 0)
    def test_hold_stays_zero_and_conditional_new_item(self):
        data=row(hold='True', new_item_if_confirmed_target='23.0')
        r=engine.apply_precomputed_policy(data,10)
        self.assertTrue(r['hold'])
        self.assertEqual(r['order_qty'],0)
        self.assertEqual(r['if_confirmed_order_qty'],13)
    def test_invalid_negative_stock(self):
        with self.assertRaises(ValueError):
            engine.apply_precomputed_policy(row(),-1)
    def test_csv_reader_and_lookup(self):
        with tempfile.TemporaryDirectory() as d:
            path=pathlib.Path(d)/'v4_inventory_policy_2015_12_31.csv'
            pd.DataFrame([row()]).to_csv(path,index=False)
            with mock.patch.object(bridge,'OFFLINE_POLICY_FILE',path):
                bridge.load_offline_policy.cache_clear()
                result=bridge.offline_v4_recommendation('FOODS_1_001','CA_1',10,5)
                self.assertEqual(result['order_qty'],3)
                self.assertEqual(result['confidence'],'Low')
                with self.assertRaises(KeyError):
                    bridge.offline_v4_recommendation('FOODS_1_002','CA_1',10)
                bridge.load_offline_policy.cache_clear()
    def test_reject_duplicate_or_wrong_date(self):
        with tempfile.TemporaryDirectory() as d:
            path=pathlib.Path(d)/'v4_inventory_policy_2015_12_31.csv'
            with mock.patch.object(bridge,'OFFLINE_POLICY_FILE',path):
                pd.DataFrame([row(),row()]).to_csv(path,index=False)
                bridge.load_offline_policy.cache_clear()
                with self.assertRaisesRegex(ValueError, 'Duplicate'):
                    bridge.load_offline_policy()
                pd.DataFrame([row(as_of='2015-12-30')]).to_csv(path,index=False)
                bridge.load_offline_policy.cache_clear()
                with self.assertRaisesRegex(ValueError, 'date'):
                    bridge.load_offline_policy()
                bridge.load_offline_policy.cache_clear()

if __name__=='__main__':unittest.main()
