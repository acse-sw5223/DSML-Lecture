from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np


class MonthEncoder(TransformerMixin, BaseEstimator):
    '''Encodes the month information from a string to
       a sin-cosin representation.
    '''

    def __init__(self):
        self.month_dic = {'Jan': 1,
                          'Feb': 2,
                          'Mar': 3,
                          'Apr': 4,
                          'May': 5,
                          'Jun': 6,
                          'Jul': 7,
                          'Aug': 8,
                          'Sep': 9,
                          'Oct': 10,
                          'Nov': 11,
                          'Dec': 12}

    def fit(self, X=None, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        months = X.month.apply(lambda x: self.month_dic.get(x))*2*np.pi/12
        X = X.drop(columns='month')
        X['sin_month'] = np.sin(months)
        X['cos_month'] = np.cos(months)

        return X[['sin_month', 'cos_month']]
