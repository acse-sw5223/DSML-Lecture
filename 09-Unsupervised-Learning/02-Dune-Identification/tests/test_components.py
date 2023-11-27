from nbresult import ChallengeResultTestCase


class TestComponents(ChallengeResultTestCase):
    def test_minimal_pc(self):
        self.assertGreaterEqual(self.result.min_pc, 27)
        self.assertLessEqual(self.result.min_pc, 36)
