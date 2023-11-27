from nbresult import ChallengeResultTestCase

class TestMonth_score(ChallengeResultTestCase):
    def test_score(self):
        self.assertGreater(self.result.score, 3.4)
        self.assertLess(self.result.score,3.5)

