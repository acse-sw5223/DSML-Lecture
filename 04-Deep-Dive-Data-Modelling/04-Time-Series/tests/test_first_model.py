from nbresult import ChallengeResultTestCase

class TestFirst_model(ChallengeResultTestCase):
    def test_score(self):
        self.assertLess(self.result.score, 1.3)
    