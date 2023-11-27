import sys
import os
from nbresult import ChallengeResultTestCase
import pandas as pd
import numpy as np
import scipy

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from river_geochemistry import GeochemPrep

class TestGeochem_prep(ChallengeResultTestCase):
    def test_is_sparse(self):
        print('did I make it')
        self.assertEqual(self.result.sparsity == scipy.sparse.csr_matrix, True,
                         msg="Your X_train is not saved as a sparse matrix")
        
    def test_can_transform_trainset(self):
        data = pd.read_csv('../raw_data/river_sediments.csv')
        pipe = self.result.pipeline
        print(type(pipe))
        try:
            self.assertEqual(type(pipe.transform(data)) == scipy.sparse.csr_matrix, True,
                         msg="Your pipeline class does not return a sparse matrix")
        except:
            self.assertEqual(1,0, msg="Your class cannot transform the train set.")

    def test_can_transform_testset(self):
        data = pd.read_csv('river_sediments_test.csv')
        pipe = self.result.pipeline

        try:
            self.assertEqual(type(pipe.transform(data)) == scipy.sparse.csr_matrix, True,
                         msg="Your pipeline class does not return a sparse matrix")
        except:
            self.assertEqual(1,0, msg="Your class cannot transform an unknown test set - did you use handle_unknown='ignore' in your OneHotencoder?")