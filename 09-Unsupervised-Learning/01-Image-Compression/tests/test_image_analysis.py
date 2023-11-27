from nbresult import ChallengeResultTestCase

class TestImageAnalysis(ChallengeResultTestCase):
    def test_img_shape(self):
        self.assertEqual(self.result.img_shape, (400, 345, 3))
    def test_color_count(self):
        self.assertEqual(self.result.color_count, 54095)
