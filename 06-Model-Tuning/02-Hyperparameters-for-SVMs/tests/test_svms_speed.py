from nbresult import ChallengeResultTestCase

class TestSvms_speed(ChallengeResultTestCase):
    def test_speed(self):
        self.assertEqual(self.result.fast_algorithm, 'sgd')
