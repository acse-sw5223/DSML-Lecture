from nbresult import ChallengeResultTestCase


class TestScaling(ChallengeResultTestCase):
    def test_no2(self):
        self.assertAlmostEqual(self.result.dataset['no2'].mean(), 0)

    def test_so2(self):
        self.assertAlmostEqual(self.result.dataset['so2'].mean(), 0)
