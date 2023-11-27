import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsRegressor

class NutritionRecommender:
    def __init__(self):
        self.data = pd.read_csv('raw_data/food.csv')
        self.X_prep = self.prepare_features()
        self.recommender = KNeighborsRegressor().fit(self.X_prep, self.X_prep['Data.Alpha Carotene'])

        return None

    def prepare_features(self):
        selector = make_column_selector(pattern='Data.Household Weights', dtype_exclude=object)
        features = self.data.select_dtypes(include=np.number).drop(columns=selector(self.data)).drop(columns='Nutrient Data Bank Number').columns

        prep_pipe = make_pipeline(FunctionTransformer(lambda x:np.log(x+0.000001)),
                         MinMaxScaler())
        preproc = make_column_transformer((prep_pipe, features))

        return pd.DataFrame(preproc.fit_transform(self.data), columns=features)


    def description_contains(self, substring:str):
        similars = []
        for desc in self.data.Description.values:
            if substring.upper() in desc.upper():
                similars.append(desc)
        return self.data[self.data.Description.isin(similars)]

    def find_similar(self, item, nb_recommendations=5):
        if type(item) == int:
            idx = item
        elif type(item) == str:
            similars = self.description_contains(item)
            if len(similars) == 0:
                print('No similar item')
                return None
            else:
                idx = similars.head(1).index[0]
        else:
            print('Unable to retrieve similar item')
            return None
        
        food = self.X_prep.iloc[idx]
        distance, neighbors = self.recommender.kneighbors([food],n_neighbors=nb_recommendations+1)
        
        return self.data.loc[neighbors[0][1:],['Category', 'Description', 'Data.Kilocalories']]
