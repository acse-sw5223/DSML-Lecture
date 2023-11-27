from nbresult import ChallengeResultTestCase


class TestStacking_score(ChallengeResultTestCase):
    def test_voting(self):
        score = self.result.voting_score
        self.assertGreaterEqual(score, .60, "You should be able to achieve close to 60% accuracy with a voting classifier")

    def test_stacking(self):
        score = self.result.stacking_score
        self.assertGreaterEqual(score, .66, "You should be able to achieve >66% accuracy with a stacked classifier")
    