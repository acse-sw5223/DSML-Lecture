from nbresult import ChallengeResultTestCase
import pandas as pd

class TestTrees(ChallengeResultTestCase):
    def test_trees_score(self):
        self.assertGreater(self.result.score, 0.83)
        self.assertLess(self.result.score,0.85)
    def test_feature_importance_format(self):
        df = self.result.importance
        self.assertTrue(type(df) == pd.Series, "Your feature importane is not a Pandas Series")
    def test_feature_importance_sorted(self):
        df = self.result.importance
        self.assertGreater(df.iloc[0], df.iloc[1], "Your feature importane is not sorted in descending order")
        

