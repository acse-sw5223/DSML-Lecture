from nbresult import ChallengeResultTestCase

class TestReduced_complexity_model(ChallengeResultTestCase):
    def test_reduced_complexity_score(self):
        self.assertEqual(abs(self.result.model_score) >= 0.37, True)

