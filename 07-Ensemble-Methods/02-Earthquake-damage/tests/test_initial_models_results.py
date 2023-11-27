from nbresult import ChallengeResultTestCase


class TestInitial_models_results(ChallengeResultTestCase):
    def test_dummy_score(self):
        score = self.result.dummy_score
        self.assertGreaterEqual(score, .44)
        self.assertLess(score, .5)

    def test_linear_score(self):
        score = self.result.linear_score
        self.assertGreaterEqual(score, .55)
        self.assertLess(score, .61)
    
    def test_knn_score(self):
        score = self.result.knn_score
        self.assertGreaterEqual(score, .55)
        self.assertLess(score, .6)

    def test_sgd_score(self):
        score = self.result.sgd_score
        self.assertGreaterEqual(score, .55)
        self.assertLess(score, .6)
    
    def test_tree_score(self):
        score = self.result.tree_score
        self.assertGreaterEqual(score, .57)
        self.assertLess(score, .61)
