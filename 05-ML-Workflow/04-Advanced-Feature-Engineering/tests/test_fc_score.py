from nbresult import ChallengeResultTestCase

class TestFc_score(ChallengeResultTestCase):
    def test_score(self):
        self.assertGreater(self.result.score, 6.8)
        self.assertLess(self.result.score,7.0)

