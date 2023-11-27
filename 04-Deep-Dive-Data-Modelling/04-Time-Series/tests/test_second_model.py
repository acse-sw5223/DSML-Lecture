from nbresult import ChallengeResultTestCase

class TestSecond_model(ChallengeResultTestCase):
    def test_score(self):
        self.assertLess(self.result.score, 1.4)
        self.assertGreater(self.result.score, 1.30)
    