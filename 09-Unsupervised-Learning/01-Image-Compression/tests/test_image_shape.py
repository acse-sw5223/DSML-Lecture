from nbresult import ChallengeResultTestCase
import numpy as np


class TestImageShape(ChallengeResultTestCase):
    def test_img_shape(self):
        self.assertEqual(self.result.image.shape, (138000, 3))
    def test_nb_colors(self):
        self.assertEqual(len(np.unique(self.result.image, axis=0)), 32)
