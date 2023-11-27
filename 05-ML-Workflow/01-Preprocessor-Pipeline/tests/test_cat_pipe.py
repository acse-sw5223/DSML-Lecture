from nbresult import ChallengeResultTestCase

class TestCat_pipe(ChallengeResultTestCase):
    def test_nb_columns(self):
        self.assertEqual(self.result.X_test.shape[1] ==1887, True)
    def test_nb_samples(self):
        self.assertEqual(self.result.X_test.shape[0] ==887, True)

