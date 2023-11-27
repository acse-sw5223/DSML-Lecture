from nbresult import ChallengeResultTestCase

class TestBaseline_score(ChallengeResultTestCase):
    def test_score(self):
        self.assertGreater(self.result.score, 4.0)
        self.assertLess(self.result.score,4.2)

