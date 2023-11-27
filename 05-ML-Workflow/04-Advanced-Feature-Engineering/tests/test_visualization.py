from nbresult import ChallengeResultTestCase

class TestVisualization(ChallengeResultTestCase):
    def test_month(self):
        self.assertEqual(self.result.df.month.unique().shape[0] ==1, True)
        self.assertEqual(self.result.df.month.unique()[0] =='May', True)
    def test_year(self):
        self.assertEqual(self.result.df.year.unique().shape[0] ==1, True)
        self.assertEqual(self.result.df.year.unique()[0] ==2004, True)

