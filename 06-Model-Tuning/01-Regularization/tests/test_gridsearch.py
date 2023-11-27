
from nbresult import ChallengeResultTestCase


class TestGridsearch(ChallengeResultTestCase):
    def test_score(self):
        self.assertGreater(self.result.score,0.929, msg = "Your accuracy score seems a bit lower then expected.")
    def test_l1_ratio(self):
        ratio = self.result.params['l1_ratio']
        self.assertTrue(ratio==0.9,msg=f"The expected best l1_ratio is not {ratio}")