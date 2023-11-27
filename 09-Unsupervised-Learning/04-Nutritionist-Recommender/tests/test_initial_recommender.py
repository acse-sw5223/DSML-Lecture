from nbresult import ChallengeResultTestCase


class TestInitial_recommender(ChallengeResultTestCase):
    def test_number_of_recommendation(self):
        self.assertTrue(self.result.nb_recommendations,5)

    def test_first_recommendation(self):
        rec = self.result.first_recommendation
        self.assertEqual(rec,'CHEESE,GRUYERE')

    def test_second_recommendation(self):
        rec = self.result.second_recommendation
        self.assertEqual(rec,'CHEESE,LIMBURGER')

