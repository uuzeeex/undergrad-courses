# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 12:11:03 2017

@author: lamwa
"""

import numpy as np

class DataProcessor(object):
    
    def __init__(self, filename):
        
        self.filename = filename
        self.data = np.genfromtxt(self.filename)[1 :]
        self.data_size = len(self.data)
        
        self.X = self.data[:, : -1]
        self.y = self.data[:, -1]
        random_order = np.random.permutation(self.data_size)
        self.X = self.X[random_order]
        self.y = self.y[random_order]
    
    def get_data(self, train_size=50):
        self.train_X = self.X[: train_size]
        self.train_y = self.y[: train_size]
        self.test_X = self.X[train_size :]
        self.test_y = self.y[train_size :]
        return self.train_X, self.train_y, self.test_X, self.test_y
    
    def get_folds(self, batch_num=10):
        batch_size = round(self.data_size / batch_num)
        X_folds = [self.X[batch_size * i : min(batch_size * (i + 1), self.data_size)] for i in range(batch_num)]
        y_folds = [self.y[batch_size * i : min(batch_size * (i + 1), self.data_size)] for i in range(batch_num)]
        return X_folds, y_folds