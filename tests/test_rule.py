from unittest import TestCase
from niaclass.rule import _Rule


class RuleTestCase(TestCase):
    def test_init_works_fine(self):
        rule = _Rule(value="val1")
        self.assertEqual(rule.value, "val1")
        self.assertEqual(rule.min, None)
        self.assertEqual(rule.max, None)

        rule = _Rule(min_val=0.0, max_val=5.5)
        self.assertEqual(rule.value, None)
        self.assertEqual(rule.min, 0.0)
        self.assertEqual(rule.max, 5.5)
