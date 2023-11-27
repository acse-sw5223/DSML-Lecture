from nbresult import ChallengeResultTestCase

class TestF1_diff(ChallengeResultTestCase):
    def test_f1_diff(self):
        self.assertEqual(((self.result.score >= 0.13) & (self.result.score < 0.16)), True)
