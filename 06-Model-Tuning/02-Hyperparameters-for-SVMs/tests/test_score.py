from nbresult import ChallengeResultTestCase


class TestScore(ChallengeResultTestCase):
    def test_score(self):
        self.assertTrue(self.result.score > 0.84)
