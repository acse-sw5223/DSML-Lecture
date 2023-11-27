from nbresult import ChallengeResultTestCase

class TestCheck_data(ChallengeResultTestCase):
    def test_numeric(self):
        sentence = self.result.sentence
        self.assertFalse('3' in sentence, msg="You have not removed all the numerical values")
    def test_lowercase(self):
        sentence = self.result.sentence
        self.assertTrue(sentence == sentence.lower(), 
                        msg="You appear to not have transformed all your sentence to lowercase")
    def test_length(self):
        sentence = self.result.sentence
        self.assertTrue(len(sentence) == 434, 
                        msg="The length of your sentence was expected to be 434 characters.")


        