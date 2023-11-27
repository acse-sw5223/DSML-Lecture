from nbresult import ChallengeResultTestCase


class TestBase_accuracy(ChallengeResultTestCase):
    def test_gxb_score(self):
        score = self.result.score
        self.assertGreaterEqual(score, .7)
