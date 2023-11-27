from nbresult import ChallengeResultTestCase


class TestCross_validate(ChallengeResultTestCase):
    def test_cv_score(self):
        self.assertLess(self.result.cv_lr['test_score'].mean(), .2)

    def test_is_good_score(self):
        self.assertFalse(self.result.is_good)

        
