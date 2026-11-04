import pathlib
import sys
import unittest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from inventory_engine import protection_period_order, repeat_week_scenario

class TestInventoryEngine(unittest.TestCase):
    def test_scenario(self):
        forecast = repeat_week_scenario([1,2,3,4,5,6,7])
        self.assertEqual(len(forecast), 14)
        out = protection_period_order(forecast, on_hand=10, on_order=5, manual_safety_stock=4)
        self.assertEqual(out.expected_demand, 56)
        self.assertEqual(out.computed_order_qty, 45)
        self.assertEqual(out.recommended_order_qty, 45)
    def test_empirical_safety(self):
        out = protection_period_order([2]*14, on_hand=20, historical_positive_errors=[0,1,2,3,4,5,6,7,8,9])
        self.assertAlmostEqual(out.safety_stock, 8.55)
        self.assertEqual(out.computed_order_qty, 17)
    def test_hold(self):
        out = protection_period_order([1]*14, on_hand=0, manual_safety_stock=3, hold_flags=['NEW_ITEM'])
        self.assertEqual(out.computed_order_qty, 17)
        self.assertEqual(out.recommended_order_qty, 0)
        self.assertTrue(out.hold)
    def test_missing_safety(self):
        out = protection_period_order([1]*14, on_hand=0)
        self.assertIsNone(out.computed_order_qty)
        self.assertTrue(out.hold)
    def test_not_enough_days(self):
        with self.assertRaises(ValueError):
            protection_period_order([1]*7, on_hand=0)
if __name__ == '__main__': unittest.main()
