from nbresult import ChallengeResultTestCase


class TestEncoding(ChallengeResultTestCase):
    def test_columns(self):
        self.assertEqual(len(self.result.dataset.columns), 14)

    def test_max_values(self):
        data = self.result.dataset
        for feature in ['H','X','R']:
            self.assertEqual(data[feature].max(), 1)
    
    def test_number_of_features(self):
        data = self.result.dataset
        nb_features = data[['H','X','R']].sum().sum()
        self.assertEqual(nb_features, len(data))
