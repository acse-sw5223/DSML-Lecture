from nbresult import ChallengeResultTestCase

class TestEncoding(ChallengeResultTestCase):
    def test_df_shape(self):
        self.assertEqual(self.result.dataset.shape[1], 62, "The shape of your dataframe is not correct: check that you have dropped all the necessary columns")
    def test_yd(self):
        self.assertEqual(self.result.dataset.yd.max(), 1)
    def test_eh(self):
        self.assertEqual(self.result.dataset.eh.min(), 0)
    def test_removed_columns(self):
        df = self.result.dataset
        categorical_features = df.dtypes[df.dtypes==object].index.shape[0]
        self.assertEqual(categorical_features, 0, "Some categorical features were either not encoded or not removed")

