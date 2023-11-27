from nbresult import ChallengeResultTestCase


class TestClassification(ChallengeResultTestCase):
    def test_scores(self):
        score_original = self.result.full_score
        score_pca = self.result.pca_score
        self.assertLessEqual(score_original, .92, "Your full X_train score is too high")
        self.assertGreaterEqual(score_pca, .9, "Your PCA score should be higher")
    def test_speed(self):
        time_ratio = self.result.ratio
        self.assertGreaterEqual(time_ratio, 100, "PCA training should be at least 2 orders of magnitude faster than training with the full dataset")