
from nbresult import ChallengeResultTestCase


class TestLasso(ChallengeResultTestCase):
    def test_zero_impact_features(self):
        self.assertTrue(len(self.result.zero_impact_features) == 28, "Have you regularized enough? Notice that C = 1/alpha")
    def test_l1_score(self):
        self.assertTrue((self.result.score > 0.9 and self.result.score < 0.93) == True)
