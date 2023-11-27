from sklearn.base import TransformerMixin, BaseEstimator
# Your imports first

class CityEncoder(TransformerMixin, BaseEstimator):
    '''Encodes the City information as "City_Country" and deletes the "Country" column.'''

    pass


class CoordinatesFeatureCross(TransformerMixin, BaseEstimator):
    '''CoordinateFeatureCross creates a feature cross of Latitude and Longitude.
    It does so at 1 degree resolution for all coordinates on Earth.''' 
    
    pass