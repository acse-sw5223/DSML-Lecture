import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression


class GoldenPlainModel:
    def __init__(self):
        return None
    
    def train(self, path):
        df = pd.read_csv(path)
        df.drop_duplicates(inplace=True)

        df.drop(columns=['Comments', 'Comments2', 'RoadType', 'LRoadWidth', 'RRoadWidth',
       'AdjoiningL', 'OverallFue', 'GroundFuel', 'ElevatedFu',
       'BarkHazard', 'Majorweeds', 'LandForm00', 'EVCSource', 'Origin',
       'Recommenda'], inplace=True)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(df.drop(columns='RCACTreesS'),df.RCACTreesS, train_size=.8, random_state=42)
        
        self.numerical_columns = self.X_train.select_dtypes(exclude=['object']).columns.to_list()
        self.numerical_columns.remove('Powerline')
        self.numerical_columns.remove('WidthVarie')

        self.impute()
        self.scale()
        self.X_train = self.encode()

        self.features = ['RCACRegene', 'Regenerati', 'RCACMaxFau', 'RCACWildli', 'Shrubs',
       'SoilTypeNA', 'PowerlineD', 'Trees', 'CanopyCont', 'yk',
       'RCACSiteDi', 'Disturbanc']
        
        self.model = LinearRegression().fit(self.X_train[self.features], self.y_train)

    
    def evaluate(self):
        self.X_test[['PowerlineD','Trees','RoadWidthM']]= self.cw_imputer.transform(self.X_test[['PowerlineD','Trees','RoadWidthM']])
        self.impute_missing_categories(self.X_test)
        self.X_test['CanopyCont'] = self.impute_canopy(self.X_test.CanopyCont)
        self.X_test[self.numerical_columns] = self.scaler.transform(self.X_test[self.numerical_columns])
        self.X_test = self.apply_encoding(self.X_test)

        y_pred = self.model.predict(self.X_test[self.features])
        pd.DataFrame(y_pred,columns=['RCACTreesS']).to_csv('predictions.csv', index=False)

        return self.model.score(self.X_test[self.features],self.y_test)

    def impute(self):
        self.cw_imputer = SimpleImputer(strategy='most_frequent').fit(self.X_train[['PowerlineD','Trees','RoadWidthM']])
        self.X_train[['PowerlineD','Trees','RoadWidthM']]= self.cw_imputer.transform(self.X_train[['PowerlineD','Trees','RoadWidthM']])
        self.impute_missing_categories(self.X_train)
        self.X_train['CanopyCont'] = self.impute_canopy(self.X_train.CanopyCont)
    
    def impute_missing_categories(self, X):
        X.loc[X.Locality.isnull(),'Locality'] = 'not known'
        X.loc[X.EVCNotes.isnull(),'EVCNotes'] = 'no notes'
        X.loc[X.LandFormLS.isnull(),'LandFormLS'] = 'LandFormLSNA'
        X.loc[X.SoilType.isnull(),'SoilType'] = 'SoilTypeNA'

    
    def impute_canopy(self, X):
        # Create a dictionaty to replace strings by int:
        dic = {'none':0,
        'sparse':1,
        'patchy':2,
        'continuous':3,
        'c':3}

        # Use an apply lambda function to replace the value. Cast to int, and return default value as 'x'
        return X.apply(lambda x:int(dic.get(x, x)))
    
    def scale(self):
        self.scaler = RobustScaler().fit(self.X_train[self.numerical_columns])
        self.X_train[self.numerical_columns] = self.scaler.transform(self.X_train[self.numerical_columns])

    def encode(self):
        self.encoder = OneHotEncoder().fit(self.X_train[['AdjoiningV', 'SoilType']])
        return self.apply_encoding(self.X_train)

    def apply_encoding(self, X):
        ohe = self.encoder
        # horizontal stack of the two arrays
        columns = np.hstack(ohe.categories_)

        # Create an array of one hot encoded values
        ohe_df = pd.DataFrame(ohe.transform(X[['AdjoiningV', 'SoilType']]).toarray(), columns=columns)

        # Reset indexes so both dfs have the same (important: drop=True to avoid keeping index as a column)
        X.reset_index(inplace=True, drop=True)
        ohe_df.reset_index(inplace=True, drop=True)

        # Join the one hot encoded values to the original dataframe
        X = X.join(ohe_df)

        # Drop ALL original categorical variables from the dataframe and check it visually
        X.drop(['AdjoiningV', 'SoilType'], axis=1, inplace=True)

        return X
