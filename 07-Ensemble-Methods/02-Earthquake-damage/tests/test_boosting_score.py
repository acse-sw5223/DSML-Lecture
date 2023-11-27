from nbresult import ChallengeResultTestCase


class TestBoosting_score(ChallengeResultTestCase):
    def test_random_forest(self):
        score = self.result.rf_score
        self.assertGreaterEqual(score, .67, "You should be able to achieve > 67% accuracy with RandomForest")

    def test_boosted_knns(self):
        score = self.result.knn_bag_score
        self.assertGreaterEqual(score, .62, "You should be able to achieve >62% accuracy with bagged KNNs")
    