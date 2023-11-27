from nbresult import ChallengeResultTestCase
from sklearn.metrics import recall_score, precision_score
import pandas as pd


class TestFinal_classifier(ChallengeResultTestCase):
    def test_precision(self):
        y_pred = self.result.predictions
        y_true = pd.read_csv('swarm_test_labels.csv')
        self.assertGreaterEqual(precision_score(y_true, y_pred), .90, "You can achieve a precision greater than 90% [my best precision score is at 93.8%]")

    def test_recall(self):
        y_pred = self.result.predictions
        y_true = pd.read_csv('swarm_test_labels.csv')
        self.assertGreater(recall_score(y_true, y_pred), .7, "You need to achieve a recall greater than 70%")
