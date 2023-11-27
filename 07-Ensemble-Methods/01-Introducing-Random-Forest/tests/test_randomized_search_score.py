from nbresult import ChallengeResultTestCase

class TestRandomized_search_score(ChallengeResultTestCase):
    def test_recall_score(self):
        test_recall = (self.result.rf_recall > 0.89 and self.result.rf_recall < 0.92)
        self.assertTrue(test_recall, msg="I am expecting a recall around 0.90")

    def test_precision_score(self):
        test_precision = (self.result.rf_precision > 0.95 and self.result.rf_precision < 0.98)
        self.assertTrue(test_precision, msg="I am expecting a precision between 0.96-0.97")


