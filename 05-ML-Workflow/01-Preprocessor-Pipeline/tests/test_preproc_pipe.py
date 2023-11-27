from nbresult import ChallengeResultTestCase

class TestPreproc_pipe(ChallengeResultTestCase):
    def test_nb_columns(self):
        self.assertEqual(self.result.X_test.shape[1] ==1924, True)
    def test_nb_samples(self):
        self.assertEqual(self.result.X_test.shape[0] ==887, True)

