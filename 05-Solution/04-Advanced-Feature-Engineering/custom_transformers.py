
from sklearn.base import TransformerMixin, BaseEstimator
# Additional Imports:
from sklearn.feature_extraction import FeatureHasher
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np

class CityEncoder(TransformerMixin, BaseEstimator):
    '''Encodes the City information as "City_Country" and deletes the "Country" column.'''

    def __init__(self):
        pass

    def encode_cities(self, X):
        # Convert series of strings into a list of lists of strings
        return X.apply(lambda x: [f'{x.City}_{x.Country}'], axis=1).tolist()

    def fit(self, X, y=None):
        self.city_hasher = FeatureHasher(input_type='string')
        cities = self.encode_cities(X)
        self.city_hasher.fit(cities)

        return self
    
    def transform(self, X, y=None):
        cities = self.encode_cities(X)

        return self.city_hasher.transform(cities)


class CoordinatesFeatureCross(TransformerMixin, BaseEstimator):
    '''CoordinateFeatureCross creates a feature cross of Latitude and Longitude.
    It does so at 1 degree resolution for all coordinates on Earth.''' 
    
    def __init__(self):
        self.coordinates_ohe = OneHotEncoder()
        self.min_lon = -180
        self.min_lat = -90
        self.max_lon = 180
        self.max_lat = 91
        
        lats = np.arange(self.min_lat, self.max_lat, 1)
        lons = np.arange(self.min_lon, self.max_lon, 1)
    
        corpus = []
        for lat in lats:
            for lon in lons:
                corpus.append(f'{lat}_{lon}')
        self.coordinates_ohe.fit(pd.DataFrame(data=corpus, columns=['coordinates']))
    
    def fit(self, X=None, y=None):
        return self
    
    def encode_features(self,row):
        lat = int(row.Latitude)
        lon = int(row.Longitude)
    
        return f'{lat}_{lon}'

    def transform(self, X, y=None):
        vectors = X.apply(lambda x: self.encode_features(x), axis=1)
        vectors = self.coordinates_ohe.transform(pd.DataFrame(data=vectors, columns=['coordinates']))
    
        return vectors
