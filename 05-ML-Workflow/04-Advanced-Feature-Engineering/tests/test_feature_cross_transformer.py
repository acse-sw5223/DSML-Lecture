from nbresult import ChallengeResultTestCase
import scipy

class TestFeature_cross_transformer(ChallengeResultTestCase):
    def test_is_sparse(self):
        self.assertEqual(type(self.result.sparse_matrix) == scipy.sparse.csr_matrix, True)
    def test_shape(self):
        self.assertEqual(self.result.sparse_matrix.toarray().shape == (17656, 65160), True)

