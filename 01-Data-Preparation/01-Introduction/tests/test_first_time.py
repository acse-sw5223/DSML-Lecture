from nbresult import ChallengeResultTestCase


class TestFirst_time(ChallengeResultTestCase):
    def test_accuracy(self):
        self.assertLessEqual(self.result.score, 0.3,
        "The accuracy is too high")

    def test_best_model(self):
        self.assertEqual(self.result.model, 'svc',
        "expecting SVD as the best model")

    def test_number_of_values(self):
        self.assertEqual(len(self.result.predictions), 4,
        "Check that you have the right number of values in your predictions")

    def test_sum_of_values(self):
        self.assertEqual(sum(self.result.predictions), 14,
        "The values do not add-up to the expected sum")
        
