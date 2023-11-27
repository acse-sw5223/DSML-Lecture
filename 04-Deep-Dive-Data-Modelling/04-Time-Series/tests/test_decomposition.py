from nbresult import ChallengeResultTestCase

class TestDecomposition(ChallengeResultTestCase):
    def test_new_columns(self):
        data = self.result.data
        summed = 0
        for item in ['trend', 'season', 'residuals']:
            summed += (item in data.columns) 
        self.assertTrue(summed==3, "Check that you have put the trend, season columns and residulas in data")
    
    def test_residuals(self):
        residuals = self.result.data.residuals.mean()
        print(residuals)
        self.assertAlmostEqual(residuals,0.0004326930811312141)

