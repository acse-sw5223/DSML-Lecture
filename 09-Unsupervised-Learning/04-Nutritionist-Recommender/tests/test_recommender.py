from nbresult import ChallengeResultTestCase


class TestRecommender(ChallengeResultTestCase):
    def test_best_similarity(self):
        similarity = self.result.best_similarity
        self.assertEqual(sum(similarity.index == [107, 104, 5782, 80, 71]), 5, 'make sure to not return the searched item')

    def test_number_recommendation(self):
        similarity = self.result.best_similarity
        self.assertEqual(similarity.shape, (5,3), 'make sure to only return the recommended columns')
