from nbresult import ChallengeResultTestCase
import numpy as np

class TestScaling(ChallengeResultTestCase):
    def test_nan(self):
        self.assertEqual(np.isnan(self.result.test).sum(), 0)
        self.assertEqual(np.isnan(self.result.train).sum(), 0)
        self.assertEqual(np.isnan(self.result.val).sum(), 0)
    def test_distribution(self):
        self.assertLess(self.result.test.mean(), 0.01)
        self.assertLess(self.result.train.mean(), 0.01)
        self.assertLess(self.result.val.mean(), 0.01)
