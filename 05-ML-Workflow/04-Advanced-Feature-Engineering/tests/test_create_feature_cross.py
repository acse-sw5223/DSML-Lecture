from nbresult import ChallengeResultTestCase
import scipy

class TestCreate_feature_cross(ChallengeResultTestCase):
    def test_is_sparse(self):
        self.assertEqual(type(self.result.sparse_matrix) == scipy.sparse.csr_matrix, True)
    def test_shape(self):
        self.assertEqual(self.result.sparse_matrix.toarray().shape == (2, 65160), True)

