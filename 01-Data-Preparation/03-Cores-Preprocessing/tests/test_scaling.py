from nbresult import ChallengeResultTestCase


class TestScaling(ChallengeResultTestCase):
    def test_depth(self):
        self.assertEqual(int(self.result.dataset['Depth CSF-A (m)'].max()), 1)

    def test_reflectance_L(self):
        self.assertEqual(int(self.result.dataset['Reflectance L*'].mean()), 0)

    def test_reflectance_a_b(self):
        self.assertAlmostEqual(
            self.result.dataset[[
                'Reflectance a*',
                'Reflectance b*']].median().sum(), 0)
