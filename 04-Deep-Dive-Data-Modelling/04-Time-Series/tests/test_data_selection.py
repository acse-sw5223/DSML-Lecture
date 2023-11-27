from nbresult import ChallengeResultTestCase
import numpy as np

class TestData_selection(ChallengeResultTestCase):
    def test_shape(self):
        data = self.result.data
        self.assertTrue(data.shape==(3131, 7), "Check that you have removed the missing values as instructed")
    
    def test_dt_encoding(self):
        dt = self.result.data.dt
        self.assertTrue(type(dt.values[0])==np.datetime64, "Make sure to change the dt column into np.datetime64")

