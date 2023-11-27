from nbresult import ChallengeResultTestCase


class TestBest_k(ChallengeResultTestCase):
    def test_best_k(self):
        self.assertTrue(self.result.best_k < 11)
        self.assertTrue(self.result.best_k > 7)
