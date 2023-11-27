from nbresult import ChallengeResultTestCase

class TestFullPipeline(ChallengeResultTestCase):
    def test_score(self):
        score = self.result.best_accuracy
        self.assertGreaterEqual(score, .98, "Your accuracy should be greater than 98%")
        