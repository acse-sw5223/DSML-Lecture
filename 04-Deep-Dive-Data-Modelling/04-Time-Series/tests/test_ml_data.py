from nbresult import ChallengeResultTestCase
import math

class TestMl_data(ChallengeResultTestCase):
    def test_split(self):
        train_set = self.result.train_set.dt.max()
        test_set = self.result.test_set.dt.min()
        self.assertTrue(train_set<test_set, "You have not properly split your data along the date axis")
    
    def test_fraction(self):
        train = self.result.train_set
        test = self.result.test_set
        ratio = train.shape[0]/(train.shape[0]+test.shape[0])
        print(ratio)
        self.assertTrue(math.isclose(ratio,0.7,rel_tol=.05), "You have not set 70% of your data for the train set")

    def test_nb_features(self):
        nb_cols = self.result.X_train.shape[1]
        self.assertTrue(nb_cols==11, "You should have 11 features in your X matrix")

