from nbresult import ChallengeResultTestCase

class TestNum_coordinates(ChallengeResultTestCase):
    def test_score(self):
        self.assertGreater(self.result.score, 9.1)
        self.assertLess(self.result.score,9.2)

