from nbresult import ChallengeResultTestCase

class TestMonth_encoding(ChallengeResultTestCase):
    def test_sin_cos_columns(self):
        X_train_cols = self.result.train_cols
        X_test_cols = self.result.test_cols
        error_msg = "Check that you have added the 'sin_month' and the 'cos_month' columns to your "
        self.assertTrue(('sin_month' in X_train_cols) & ('cos_month' in X_train_cols),f'{error_msg}{"X_train"}')
        self.assertTrue(('sin_month' in X_test_cols) & ('cos_month' in X_test_cols),f'{error_msg}{"X_test"}')
    def test_first_month(self):
        X_train = self.result.first_data
        self.assertAlmostEqual(X_train['sin_month'], 0.84147, True)
        self.assertAlmostEqual(X_train['cos_month'], 0.54030, True)

