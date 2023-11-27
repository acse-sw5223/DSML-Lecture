from nbresult import ChallengeResultTestCase

class TestEncoder(ChallengeResultTestCase):
    def test_number_categories(self):
        encoder = self.result.encoder
        self.assertEqual(encoder.categories_[0].shape[0] ==65160, True)


