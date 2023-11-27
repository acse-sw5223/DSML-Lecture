from nbresult import ChallengeResultTestCase


class TestTarget_encoding(ChallengeResultTestCase):
    def test_columns(self):
        self.assertEqual(len(self.result.dataset.columns), 35)
    
    def test_nb_positive_class(self):
        self.assertEqual(self.result.dataset.y.sum(),225)