from nbresult import ChallengeResultTestCase

class TestVisualization(ChallengeResultTestCase):
    def test_max_diff(self):
        self.assertGreater(self.result.df.data.max(),5)
    def test_min_diff(self):
        self.assertLess(self.result.df.data.min(),-10)

