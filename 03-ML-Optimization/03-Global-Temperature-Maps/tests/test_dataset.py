from nbresult import ChallengeResultTestCase

class TestDataset(ChallengeResultTestCase):
    def test_december_2004(self):
        df = self.result.december_2004
        self.assertTrue(df.year.unique().shape[0]==1, "Check that you only selected the year 2004")
        self.assertTrue(df.shape[0]==1573, "Check that you have only selected cities in Eurasia")
    
    def test_december_2010(self):
        df = self.result.december_2010
        self.assertTrue(df.year.unique().shape[0]==1, "Check that you only selected the year 2010")
        self.assertTrue(df.shape[0]==1573, "Check that you have only selected cities in Eurasia")

