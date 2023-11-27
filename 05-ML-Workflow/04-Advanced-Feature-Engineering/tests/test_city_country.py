from nbresult import ChallengeResultTestCase
import pandas as pd
import scipy

class TestCity_country(ChallengeResultTestCase):
    def test_encoding(self):
        self.assertEqual(self.result.first_city == 'Kingston_Jamaica', True)
    def test_shape(self):
        self.assertEqual(self.result.X_train.shape == (84184, 8), True)
    

