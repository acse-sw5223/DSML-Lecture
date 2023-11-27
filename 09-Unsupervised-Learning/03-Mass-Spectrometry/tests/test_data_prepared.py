from nbresult import ChallengeResultTestCase
import numpy as np


class TestData_prepared(ChallengeResultTestCase):
    def test_X_shape(self):
        df = self.result.data
        self.assertEqual(df.shape, (362, 6))

    def test_X_scaled(self):
        df = self.result.data
        self.assertEqual(df.min().sum(), 0)
        self.assertEqual(df.max().iloc[0], 1)

    def test_y_encoding(self):
        df = self.result.label
        self.assertEqual(df.dtype, np.int64)
