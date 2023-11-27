from nbresult import ChallengeResultTestCase
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import numpy as np
import pandas as pd
import warnings


class TestFinal_results(ChallengeResultTestCase):
    def test_precision(self):
        precision = np.mean(precision_score(self.get_labels(), self.result.predictions, average=None))
        self.assertEqual(((precision >= 0.88) & (precision < 0.92)), True)
    def test_recall(self):
        recall = np.mean(recall_score(self.get_labels(), self.result.predictions, average=None))
        self.assertEqual(((recall >= 0.88) & (recall < 0.92)), True)
    def test_f1(self):
        f1 = np.mean(f1_score(self.get_labels(), self.result.predictions, average=None))
        self.assertEqual(((f1 >= 0.88) & (f1 < 0.92)), True)
    def test_accuracy(self):
        accuracy = np.mean(accuracy_score(self.get_labels(), self.result.predictions))
        self.assertEqual(((accuracy >= 0.88) & (accuracy < 0.92)), True)
    def get_labels(self):
        warnings.filterwarnings("ignore")
        try:
            if self.labels.shape[0] == 2000:
                return self.labels
        except:
            labels = self.result.data
            labels = self.result.encoder.transform(labels.values)
            self.labels = np.squeeze(labels)
            return self.labels


