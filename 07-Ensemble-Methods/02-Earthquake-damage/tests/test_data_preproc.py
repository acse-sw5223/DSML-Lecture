from nbresult import ChallengeResultTestCase
import scipy


class TestData_preproc(ChallengeResultTestCase):
    def test_data_shape(self):
        X_train_shape = self.result.X_train_shape
        X_test_shape = self.result.X_test_shape

        self.assertEqual(X_train_shape, (20848, 69), "X_train shape incorrect")
        self.assertEqual(X_test_shape, (5212, 69), "X_test shape incorrect")

    def test_is_sparse(self):
        sparsity = self.result.sparsity
        self.assertEqual(sparsity, True , "Your X_test is not a sparse matrix")