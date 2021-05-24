from unittest import TestCase
from niaclass.feature_info import _FeatureInfo


class FeatureInfoTestCase(TestCase):
    def test_init_works_fine(self):
        vals = ["val1", "val2", "val3"]
        f_info = _FeatureInfo(dtype=0, values=vals)
        self.assertEqual(f_info.dtype, 0)
        self.assertEqual(f_info.min, None)
        self.assertEqual(f_info.max, None)
        self.assertEqual(f_info.values, vals)

        f_info = _FeatureInfo(dtype=1, min_val=0.0, max_val=5.5)
        self.assertEqual(f_info.dtype, 1)
        self.assertEqual(f_info.min, 0.0)
        self.assertEqual(f_info.max, 5.5)
        self.assertEqual(f_info.values, None)
