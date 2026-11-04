import unittest
from precompute_v4_inventory import apply_stock_override

class TestPolicyOverride(unittest.TestCase):
    def test_manual_stock(self):
        r={'hold':False,'order_up_to':17.4,'new_item_if_confirmed_target':''}
        self.assertEqual(apply_stock_override(r,10)['order_qty'],8)
        self.assertEqual(apply_stock_override(r,25)['order_qty'],0)
        self.assertEqual(apply_stock_override(r,10,5)['order_qty'],3)
    def test_hold(self):
        r={'hold':True,'order_up_to':17.4,'new_item_if_confirmed_target':23}
        self.assertEqual(apply_stock_override(r,10),{'order_qty':0,'hold':True,'if_confirmed_order_qty':13})
    def test_bad_stock(self):
        with self.assertRaises(ValueError):apply_stock_override({'hold':False,'order_up_to':17},-1)

if __name__=='__main__': unittest.main()
