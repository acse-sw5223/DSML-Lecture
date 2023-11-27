from nbresult import ChallengeResultTestCase


class TestDuplicates(ChallengeResultTestCase):
    def test_duplicate_count(self):
        self.assertEqual(self.result.duplicates, 2282)

    def test_dataset_length(self):
        self.assertEqual(len(self.result.dataset), 22935)
