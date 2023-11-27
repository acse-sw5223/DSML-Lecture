from nbresult import ChallengeResultTestCase

class TestModel_performance(ChallengeResultTestCase):
    def test_bow(self):
        bow = self.result.bow_model
        self.assertGreaterEqual(bow, 0.82,
                         msg="The accuracy of your bag-of-words model should be > 0.82")
    def test_best_performance(self):
        bow = self.result.bow_model
        ng = self.result.ng_model
        self.assertTrue(bow>=ng, 
                        msg="The accuracy of your N-gram model should be lower then your bag-of-words model.")
    