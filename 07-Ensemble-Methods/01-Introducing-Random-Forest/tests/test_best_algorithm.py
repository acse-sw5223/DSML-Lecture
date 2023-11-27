from nbresult import ChallengeResultTestCase
from sklearn.ensemble import RandomForestClassifier

class TestBest_algorithm(ChallengeResultTestCase):
    def test_model(self):
        target = 1
        value = 0

        if type(self.result.model) == dict:
            target = RandomForestClassifier
            value = type(list(self.result.model.values())[0])
        if type(self.result.model) == str:
            target = "RandomForestClassifier"
            value = self.result.model

        self.assertTrue(target == value)

    def test_score(self):
        score = self.result.best_score
        score_range = (score > 0.95 and score < 0.96)
        self.assertTrue(score_range, msg="I am expecting your score to be between 0.95 and 0.96")

