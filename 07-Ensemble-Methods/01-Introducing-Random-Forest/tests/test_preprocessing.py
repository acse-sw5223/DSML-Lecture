from nbresult import ChallengeResultTestCase
import pandas as pd

class TestPreprocessing(ChallengeResultTestCase):
    def test_shape(self):
        X_train = self.result.data
        self.assertEqual(X_train.shape, (8875, 8), "You do not seem to have the correct X_train shape")
    def test_mean(self):
        X_train = self.result.data
        self.assertAlmostEqual(X_train.mean().sum(), 0)
        

