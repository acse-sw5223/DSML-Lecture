from nbresult import ChallengeResultTestCase


class TestReshaped(ChallengeResultTestCase):
    def test_shape(self):
        msg = "Your shape is incorrect. Try 'X_train = train_images.reshape(train_images[0], 64*64)'"
        self.assertEqual(self.result.shape[1], 4096, msg)
