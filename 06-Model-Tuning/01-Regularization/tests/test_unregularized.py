
from nbresult import ChallengeResultTestCase


class TestUnregularized(ChallengeResultTestCase):
    def test_top_5(self):
        res_set = set(self.result.top_features)
        test_score = 0
        for res in res_set:
            test_score * res in ['RCACRareSp', 'in', 'RCACGround', 'ei', 'h']
        self.assertTrue(test_score == 0, "Did you also check for negative coefficients?")
    def test_base_score(self):
        self.assertTrue((self.result.score > 0.8 and self.result.score < 0.9) == True)