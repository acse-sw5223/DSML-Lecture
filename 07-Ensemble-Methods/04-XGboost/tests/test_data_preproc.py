from nbresult import ChallengeResultTestCase
import pandas as pd
import numpy as np


class TestData_preproc(ChallengeResultTestCase):
    def test_data_shape(self):
        self.assertEqual(self.result.X_shape, (208480, 39), "X_train shape incorrect")

    def test_y_encoding(self):
        self.assertEqual(self.result.y_values, 3 , "Your y_test does not have the right number of categories")
        self.assertEqual(self.result.y_type, np.int64 , "Your y_test is not correctly encoded")

        
