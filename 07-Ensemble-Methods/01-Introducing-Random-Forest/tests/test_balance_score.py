from nbresult import ChallengeResultTestCase


class TestBalance_score(ChallengeResultTestCase):
    def test_precision(self):
        score = self.result.precision_score
        score_test = (score < 0.97 and score > 0.95)
        self.assertTrue(score_test, 
                        msg = "I am expecting your score to be around 0.95 for precision")

    def test_recall(self):
        score = self.result.recall_score
        score_test = (score < 0.86 and score > 0.82)
        self.assertTrue(score_test, 
                        msg = "I am expecting your score to be around 0.85 for recall")
    