from nbresult import ChallengeResultTestCase

class TestDummy_model(ChallengeResultTestCase):
    def test_score(self):
        self.assertLess(self.result.score, 1.3)
        self.assertGreater(self.result.score, 1.2)
    