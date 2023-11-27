import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import cv2
import zipfile
import shutil


class DunesDataLoader:
    def __init__(self):


        self.train_set, self.test_set = self.create_splits()
    

        
    def create_splits(self):
        dunes = os.listdir('raw_data/dunes_dataset/is_dune')
        no_dunes = os.listdir('raw_data/dunes_dataset/not_dune')
    
        dunes_df = pd.DataFrame(data=dunes, columns=['image_path'])
        dunes_df['folder'] = 'raw_data/dunes_dataset/is_dune'
        dunes_df['target'] = 1
    
        not_dunes_df = pd.DataFrame(data=no_dunes, columns=['image_path'])
        not_dunes_df['folder'] = 'raw_data/dunes_dataset/not_dune'
        not_dunes_df['target'] = 0
    
        dunes_train, dunes_test = train_test_split(dunes_df, train_size=.7, random_state=42)
        not_dunes_train, not_dunes_test = train_test_split(not_dunes_df, train_size=.7, random_state=42)
    
        train_set = pd.concat([dunes_train, not_dunes_train], axis=0)
        test_set = pd.concat([dunes_test, not_dunes_test], axis=0)
    
        return shuffle(train_set,random_state=42).reset_index(), shuffle(test_set,random_state=42).reset_index()
    
    def get_data(self):
        train_data = []
        test_data = []
        
        for df, collection in zip([self.train_set, self.test_set],[train_data, test_data]):
            for i,image in df.iterrows():
                im = cv2.imread(f'{image.folder}/{image.image_path}',0)
                collection.append(im)
        
        return np.array(train_data), np.array(test_data), self.train_set.target, self.test_set.target
    
    