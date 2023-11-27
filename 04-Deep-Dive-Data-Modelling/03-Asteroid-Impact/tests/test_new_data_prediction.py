from nbresult import ChallengeResultTestCase
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

class TestNewDataPrediction(ChallengeResultTestCase):

    def test_ROC_AUC_Score(self):
        true_labels = pd.read_csv('new_samples_labels.csv')
        predicted_labels = pd.read_csv('../predictions.csv')
        performance = roc_auc_score(true_labels, predicted_labels)

        self.assertGreater(performance, 0.82, msg="Check that you have used the ROC-AUC metric - you should obtain a score > 0.82. Also, in your EDA, have you checked for class imbalance?")
    
    def test_predicted_performance(self):
        true_labels = pd.read_csv('new_samples_labels.csv')
        predicted_labels = pd.read_csv('../predictions.csv')
        performance = roc_auc_score(true_labels, predicted_labels)
        delta = np.abs(self.result.predicted_performance - performance)

        self.assertLess(delta, 0.1, msg="Your error prediction is >0.1 points away from actual. Did you use a proper test set?")

    def test_number_features(self):
        nb_features = len(self.result.X_train.columns)

        self.assertLess(nb_features, 4, msg="You seem to use a lot of features for your prediction - have you done any feature importance tests?")


