from nbresult import ChallengeResultTestCase

class TestModel_performance(ChallengeResultTestCase):
    def test_accuracy(self):
        self.assertEqual(self.result.score < 2, True, "Your root mean squared error is too high")
