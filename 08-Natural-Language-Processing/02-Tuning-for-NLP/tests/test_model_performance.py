from nbresult import ChallengeResultTestCase

class TestModel_performance(ChallengeResultTestCase):
    def test_bow(self):
        score = self.result.score
        self.assertGreaterEqual(score, 0.83,
                         msg="The accuracy of your bag-of-words model should be > 0.83")
    