from nbresult import ChallengeResultTestCase

class TestMissing_values(ChallengeResultTestCase):
    def test_number_of_columns(self):
        self.assertEqual(self.result.dataset.shape[1], 48,"Check if you have removed all of the columns with >30% missing data")
    def test_all_zeros(self):
        score = self.result.dataset.isnull().sum().max()
        self.assertEqual(score, 0, "Check if you have more missing values in df")
    def test_CanopyCont_shape(self):
        score = self.result.dataset.CanopyCont.value_counts().shape[0]
        self.assertEqual(score, 4, "Check that you have correctly encoded CanopyCount: you should have only 0,1,2,3 as possible values")
    def test_CanopyCont_type(self):
        score = self.result.dataset.CanopyCont.dtype
        self.assertEqual(score, 'float64', "Make sure to cast the values of CanopyCont to int64")
    def test_SoilType(self):
        score = 'SoilTypeNA' in self.result.dataset.SoilType.unique()
        self.assertEqual(score, True, "Make sure to replace missing values in SoilType by 'SoilTypeNA'")
