from nbresult import ChallengeResultTestCase


class TestTarget(ChallengeResultTestCase):
    def test_target(self):
        categories = self.result.target.unique()
        self.assertTrue(categories.shape[0] == 2)
        self.assertTrue(categories[0] in [0,1])
        self.assertTrue(categories[1] in [0,1])
