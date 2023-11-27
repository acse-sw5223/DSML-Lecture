from nbresult import ChallengeResultTestCase


class TestFitting(ChallengeResultTestCase):
    def test_fitting(self):
        self.assertEqual(self.result.performance, 'overfit')

    