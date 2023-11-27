from nbresult import ChallengeResultTestCase


class TestSplitting(ChallengeResultTestCase):
    def test_testing(self):
        self.assertEqual(self.result.test.shape[0], 20000)
        self.assertEqual(self.result.test.shape[1], 10)
    def test_training(self):
        self.assertEqual(self.result.train.shape[0], 79200)
        self.assertEqual(self.result.train.shape[1], 10)
    def test_validation(self):
        self.assertEqual(self.result.val.shape[0], 800)
        self.assertEqual(self.result.val.shape[1], 10)
