from numpy import ndarray
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import precision_recall_curve
from sklearn.base import BaseEstimator, TransformerMixin

class LogisticRegressionThreshold(LogisticRegression):

    def fit(self, X, y, sample_weight=None):
        super().fit(X, y, sample_weight)
        self.set_threshold(X,y)
        return self

    def predict(self, X) -> ndarray:
        return pd.Series(super().predict_proba(X)[:,1]).apply(lambda x: x>=self.threshold)
    
    def set_threshold(self, X, y):
        predictions_proba_baseline = self.predict_proba(X)
        precision_bl, recall_bl, threshold_bl = precision_recall_curve(y, predictions_proba_baseline[:,1],pos_label=1)
        
        prec_recall_bl_df = pd.DataFrame({'threshold':threshold_bl,
              'precision':precision_bl[:-1],
             'recall':recall_bl[:-1]})
        
        selected_recall_bl_df = prec_recall_bl_df[prec_recall_bl_df['recall']>=0.7].\
            sort_values(by='recall', ascending=True).\
            reset_index(drop=True)
        
        self.threshold =  selected_recall_bl_df.iloc[0]['threshold']


class MyTransformer(BaseEstimator, TransformerMixin):
    pass

class SwarmBehaviorPredictor:
    '''
    Can automatically load data and train a logistic regression model.
    Can then also return the predictions based on a predetermined threshold.
    '''

    def __init__(self, path, min_recall=0.7):
        self.min_recall = min_recall
        self.X_train, self.X_test, self.y_train, self.y_test = self.load_and_split(path)
        self.create_model()

    def load_and_split(self, path):
        df = pd.read_csv(path)
        df.drop_duplicates(inplace=True)

        X = df.drop(columns='Swarm_Behaviour')
        y = df.Swarm_Behaviour

        return train_test_split(X, y, test_size=0.3, random_state=42)
    
    def create_model(self):
        self.model = make_pipeline(SimpleImputer(), StandardScaler(), LogisticRegression(max_iter=5000))
        self.model.fit(self.X_train, self.y_train)
        self.set_threshold()

        return None
    
    def set_threshold(self):
        predictions_proba_baseline = self.model.predict_proba(self.X_train)
        precision_bl, recall_bl, threshold_bl = precision_recall_curve(self.y_train, predictions_proba_baseline[:,1],pos_label=1)
        
        prec_recall_bl_df = pd.DataFrame({'threshold':threshold_bl,
              'precision':precision_bl[:-1],
             'recall':recall_bl[:-1]})
        
        selected_recall_bl_df = prec_recall_bl_df[prec_recall_bl_df['recall']>=self.min_recall].\
            sort_values(by='recall', ascending=True).\
            reset_index(drop=True)
        
        self.threshold = selected_recall_bl_df.iloc[0]['threshold']

    def predict(self, X):
        '''Returns binary predictions based on probability threshold.'''

        y_proba = self.model.predict_proba(X)
        y_proba = pd.Series(y_proba[:,1], name='y_proba')

        return y_proba.apply(lambda x: x>=self.threshold)
