from nbresult import ChallengeResultTestCase

class TestNum_pipe(ChallengeResultTestCase):
    def test_nb_columns(self):
        self.assertEqual(len(self.result.X_test.columns) ==37, True)
    def test_nb_samples(self):
        self.assertEqual(self.result.X_test.shape[0] ==887, True)

