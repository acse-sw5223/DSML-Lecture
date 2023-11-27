from nbresult import ChallengeResultTestCase


class TestFinal_score(ChallengeResultTestCase):
    def test_final_accuracy(self):
        score = self.result.score
        self.assertGreaterEqual(score, .71, "You should be able to achieve > 71% accuracy")
    