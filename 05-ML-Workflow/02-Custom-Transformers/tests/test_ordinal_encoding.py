from nbresult import ChallengeResultTestCase

class TestOrdinal_encoding(ChallengeResultTestCase):
    def test_score(self):
        score = self.result.score
        self.assertTrue(((score > 3.5) & (score < 4.5)), msg="Your root mean squared error is different from expected.")

