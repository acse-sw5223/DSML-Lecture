from nbresult import ChallengeResultTestCase


class TestY_cat(ChallengeResultTestCase):

    def test_number_cat(self):
        cat_counts = self.result.dataset.shape[0]
        self.assertEqual(cat_counts, 216734)
        
