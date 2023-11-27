from nbresult import ChallengeResultTestCase
import numpy as np


class TestData_preparation(ChallengeResultTestCase):
    def test_X_prep_shape(self):
        self.assertEqual(
            self.result.data.shape, (7413, 41),
            'Double check that you have all the correct columns')

    def test_feature_engineering(self):
        data = self.result.data
        self.assertAlmostEqual(data['Data.Vitamins.Vitamin B6'].mean(), 0.6081353821297847)
        self.assertAlmostEqual(data['Data.Major Minerals.Copper'].mean(), 0.5956975128162423)

