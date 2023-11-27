from nbresult import ChallengeResultTestCase

class TestMonth_encoding(ChallengeResultTestCase):
    def test_sin_cos_columns(self):
        X_test = self.result.test_set
        error_msg = "Check that you have added the 'sin_month' and the 'cos_month' columns to your "
        self.assertTrue(('sin_month' in X_test.columns) & ('cos_month' in X_test.columns),f'{error_msg}{"X_test"}')
    def test_first_month(self):
        X_test = self.result.test_set
        self.assertAlmostEqual(X_test['sin_month'][0], 0.499, True)
        self.assertAlmostEqual(X_test['cos_month'][0], -0.866, True)
    def test_score(self):
        score = self.result.score
        self.assertLess(score, 3.5, msg="Your root mean squared error is high.")

