from nbresult import ChallengeResultTestCase


class TestCheck_df(ChallengeResultTestCase):
    def test_columns(self):
        cols = sorted(self.result.df.columns)
        desired_cols = sorted([ 'first_MZ44_Sample', 'name', 'D47', 'd13C',
       'd18O', '49_param', 'std_d13C', 'std_d18O', 'std_D47', 'd13C_diff',
       'd18O_diff', 'D47_diff','first_MZ44_Ref_Gas'])
        self.assertEqual(cols, desired_cols, "Your columns or column names are not correct")
    def test_d18O_standard_values(self):
        df = self.result.df.copy()
        standard = 'Carrara Marble'
        df = df[df.name==standard]
        print(df.shape)
        self.assertAlmostEqual(df.std_d18O.mean(), -1.82)
    def test_d13C_standard_values(self):
        df = self.result.df.copy()
        standard = 'IOL'
        df = df[df.name==standard]
        self.assertAlmostEqual(df["std_d13C"].astype(float).mean(), 0.3)
    def test_D47_standard_values(self):
        df = self.result.df.copy()
        standard = 'ETH1 carbonate'
        df = df[df.name==standard]
        self.assertAlmostEqual(df.std_D47.mean(), 0.2052)
    def test_diff_D47(self):
        df = self.result.df.copy()
        standard = 'ETH2 carbonate'
        df = df[df.name==standard]
        self.assertAlmostEqual(df.D47_diff.mean(), -0.001398550724637671)