from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_extraction import FeatureHasher
import pandas as pd
import numpy as np


class LogTransformer(TransformerMixin, BaseEstimator):
    """This class does a simple log-transform of some of the data."""

    def __init__(self, seed = 1e-5):
        self.seed=seed
        return None

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return np.log(X+self.seed)


class ZeroTransform(TransformerMixin, BaseEstimator):
    """This transformer replaces negative values by zeros: elemental concentrations cannot be < zero."""
    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = np.where(X < 0, 0, X)
        return X


class OrdinalTransformer(TransformerMixin, BaseEstimator):
    """Transforms the Ordinal categories"""

    def __init__(self, categories):
        self.categories = categories
        return None

    def fit(self, X=None, y=None):
        return self

    def transform(self, X, y=None):
        X = X.copy()
        if type(X)==pd.DataFrame:
            X = X.values
        for value, categorie in  zip(range(len(self.categories)),self.categories):
            idxs = np.where(X==categorie)[0]
            X[idxs] = value

        return X


class GeochemPrep:
    """This is the main class, it provides all the necessary transformations."""

    def make_ord_pipe(self):
        ord_encode_pipe = make_pipeline(SimpleImputer(missing_values=np.nan, strategy='most_frequent'),
                       SimpleImputer(missing_values='Undefined', strategy='most_frequent'), 
                       OrdinalTransformer(categories=['Stagnant','Slow', 'Moderate', 'Fast','Torrent','Clear', 'Colourless', 'White/Cloudy','Brown/Clear','Brown/Cloudy','Brown']),
                       StandardScaler())
        return ord_encode_pipe

    def make_hasher_pipe(self):
        # Also possible to use OneHotEncoding if handle_unknown=ignore

        hashers = []
        to_hash = ['MASTERID','NAME','STRAT','SORC','PHYS','DRNP','CONT',
                   'BANK','BNKP','COMP','SEDC','SEDP','MAT','ORDR','TYPE','TYPE2']

        for feature in to_hash:
            hashers.append((f'hashed_{feature}',make_pipeline(SimpleImputer(strategy='most_frequent'), FeatureHasher(input_type='string',n_features=1024)), [feature]))

        return ColumnTransformer(hashers)

    def fit(self, X_train, y_train=None):

        # Setting my features into distinct categories
        no_log = ['YEAR', 'LAT', 'LONG']
        to_log = ['WDTH', 'DPTH', 'Cu_AAS_PPM', 'Au_ICP_PPB', 'Ag_ICP_PPB', 'Al_ICP_PCT',
       'As_ICP_PPM', 'Ba_ICP_PPM', 'Bi_ICP_PPM', 'Ca_ICP_PCT', 'Cd_ICP_PPM',
       'Co_ICP_PPM', 'Cr_ICP_PPM', 'Fe_ICP_PCT', 'Hg_ICP_PPB', 'K_ICP_PCT',
       'Mg_ICP_PCT', 'Mn_ICP_PPM', 'Mo_ICP_PPM', 'Na_ICP_PCT', 'Ni_ICP_PPM',
       'P_ICP_PCT', 'S_ICP_PCT', 'Sb_ICP_PPM', 'Sc_ICP_PPM', 'Se_ICP_PPM',
       'Sr_ICP_PPM', 'Te_ICP_PPM', 'Th_ICP_PPM', 'Ti_ICP_PCT', 'Tl_ICP_PPM',
       'U_ICP_PPM', 'V_ICP_PPM', 'W_ICP_PPM', 'La_ICP_PPM', 'Au_INA_PPB',
       'Ba_INA_PPM', 'Br_INA_PPM', 'Ce_INA_PPM', 'Cr_INA_PPM', 'Cs_INA_PPM',
       'Hf_INA_PPM', 'Lu_INA_PPM', 'Na_INA_PCT', 'Rb_INA_PPM', 'Sc_INA_PPM',
       'Ta_INA_PPM', 'Yb_INA_PPM', 'WT_INA_g', 'pH', 'Uw_LIF_PPB',
       'Fw_ISE_PPB']
        to_ordinal = ['FLOW', 'WTRC']

        hashers_pipe = self.make_hasher_pipe()
        ord_encode_pipe = self.make_ord_pipe()
        num_log_pipe = make_pipeline(SimpleImputer(), ZeroTransform(),
                                     LogTransformer(seed=6e-3), RobustScaler())
        num_pipe = make_pipeline(SimpleImputer(), StandardScaler())

        non_hash = ColumnTransformer([('no_log_num',num_pipe,no_log),
                            ('log_num', num_log_pipe, to_log),
                            ('ordinal_features',ord_encode_pipe,to_ordinal)])

        self.final_pipe = FeatureUnion([('hashing',hashers_pipe), ('not_hashing',non_hash)])
        self.final_pipe.fit(X_train)

        return self

    def transform(self, X):
        return self.final_pipe.transform(X)
