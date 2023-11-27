from nbresult import ChallengeResultTestCase
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

class TestTransformer(ChallengeResultTestCase):
    def test_score(self):
        final_pipe = self.result.pipe
        test_set = pd.read_csv('../raw_data/test_temperatures.csv')
        train_set = pd.read_csv('../raw_data/train_temperatures.csv')

        X_train = train_set.drop(columns='AverageTemperature')
        X_test = test_set.drop(columns='AverageTemperature')

        y_train = train_set.AverageTemperature
        y_test = test_set.AverageTemperature

        final_pipe.fit(X_train, y_train)
        score = np.sqrt(mean_squared_error(y_test, final_pipe.predict(X_test)))

        self.assertTrue(((score > 3.0) & (score < 3.6)), msg="Your root mean squared error is different from expected.")

