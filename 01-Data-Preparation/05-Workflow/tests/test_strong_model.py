from nbresult import ChallengeResultTestCase

class TestStrong_model(ChallengeResultTestCase):
    def test_strong_model_score(self):
        self.assertEqual(self.result.score > 0.9, True, "Your score is too low")
    def test_nb_features(self):
        self.assertEqual(self.result.nb_features <= 7, True, "You have too many features")
