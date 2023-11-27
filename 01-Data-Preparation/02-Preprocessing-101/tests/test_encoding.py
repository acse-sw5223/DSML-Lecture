from nbresult import ChallengeResultTestCase


class TestEncoding(ChallengeResultTestCase):
    def test_columns(self):
        self.assertEqual(self.result.nb_columns, 37)

    def test_max_values(self):
        self.assertEqual(sum(self.result.max_values), 4)
    
    def test_number_of_features(self):
        self.assertEqual(self.result.nb_ohe, 34)
