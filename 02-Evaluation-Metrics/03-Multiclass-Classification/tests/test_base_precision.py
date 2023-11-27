from nbresult import ChallengeResultTestCase


class TestBase_precision(ChallengeResultTestCase):
    def test_precision_score(self):
        self.assertTrue(self.result.score < 0.9)
        self.assertTrue(self.result.score > 0.8)
    def test_dummy_score(self):
        self.assertTrue(self.result.dummy < 0.7)
        self.assertTrue(self.result.dummy > 0.6)
