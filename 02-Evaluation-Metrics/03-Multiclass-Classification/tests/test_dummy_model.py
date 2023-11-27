from nbresult import ChallengeResultTestCase

class TestDummy_model(ChallengeResultTestCase):
    def test_dummy_model_score(self):
        self.assertEqual(((self.result.score >= 0.2) & (self.result.score < 0.3)), True)
    def test_best_performer(self):
        self.assertEqual(self.result.performer, 'soil with vegetation stubble')