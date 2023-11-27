from nbresult import ChallengeResultTestCase
import pandas as pd
import scipy

class TestCity_encoding(ChallengeResultTestCase):
    def test_is_sparse(self):
        self.assertEqual(self.result.sparsity == scipy.sparse.csr_matrix, True)
    def test_shape(self):
        self.assertEqual(self.result.shape == (84184, 1048576), True)
    

