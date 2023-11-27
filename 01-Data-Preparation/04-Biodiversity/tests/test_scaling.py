from nbresult import ChallengeResultTestCase

class TestScaling(ChallengeResultTestCase):
    def test_numerical_columns(self):
        self.assertEqual(len(self.result.features), 36, "Check that you have not included Powerline and WidthVarie in your numerical columns.")
    def test_central_tendency(self):
        score = 0
        for feature in self.result.features:
            score = score + self.result.dataset[feature].median()
        self.assertEqual(score, 0, "Some of your numerical data are not properly scaled.")
