from nbresult import ChallengeResultTestCase

class TestMinSamples(ChallengeResultTestCase):
    def test_min_samples(self):
        nb_samples = self.result.min_samples
        self.assertLessEqual(nb_samples,1500)
        self.assertGreaterEqual(nb_samples,1000)
