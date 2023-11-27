
from nbresult import ChallengeResultTestCase


class TestRidge(ChallengeResultTestCase):
    def test_top_5(self):
        res_set = set(self.result.top_features)
        test_score = 0
        for res in res_set:
            test_score * res in ['RCACRareSp', 'Weeds', 'RCACRegene', 'RCACLogs', 'RCACGrassl']
        self.assertTrue(test_score == 0, "Have you regularized enough? Notice that C = 1/alpha")
    def test_l2_score(self):
        self.assertTrue((self.result.score > 0.9 and self.result.score < 0.93) == True)
    