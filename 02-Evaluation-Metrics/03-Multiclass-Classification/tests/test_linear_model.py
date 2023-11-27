from nbresult import ChallengeResultTestCase


class TestLinear_model(ChallengeResultTestCase):
    def test_best_approach(self):
        self.assertEqual(self.result.approach, 'multinomial')
    def test_ovr_accuracy(self):
        self.assertGreater(self.result.accuracy, 0.8)
