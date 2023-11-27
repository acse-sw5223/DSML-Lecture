from nbresult import ChallengeResultTestCase

class TestCheck_outliers(ChallengeResultTestCase):
    def test_nb_outliers(self):
        outliers = self.result.outliers
        self.assertEqual(outliers.shape[0], 4)
    def test_outliers_index(self):
        outliers = self.result.outliers
        index_sum = (outliers.index == [217, 218, 219, 356]).sum()
        self.assertEqual(index_sum, 4)
        