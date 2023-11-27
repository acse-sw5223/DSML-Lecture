from nbresult import ChallengeResultTestCase

class TestEncode_features(ChallengeResultTestCase):
    def test_encoded_string(self):
        self.assertEqual(self.result.encoded_string =="23_-5", True)

