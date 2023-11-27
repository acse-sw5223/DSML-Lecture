from nbresult import ChallengeResultTestCase


class TestDuplicates(ChallengeResultTestCase):
    def test_duplicate_count(self):
        self.assertEqual(self.result.duplicates, 400)

    def test_dataset_length(self):
        self.assertEqual(self.result.dataset.shape, (216741, 10))
