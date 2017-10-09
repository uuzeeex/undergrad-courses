# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 14:15:38 2017

@author: lamwa
"""

import numpy as np

class DiscriminativeModel(object):
    def fit(self, X, y):
        pass
    
    def pred(self, X):
        pass

class RidgeRegression(DiscriminativeModel):
    def __init__(self, delta_sqr=1.):
        self.delta_sqr = delta_sqr
    
    def fit(self, X, y):
        d_s = self.delta_sqr
        f_s = X.shape[1]
        self.theta = np.linalg.inv(X.T.dot(X) + d_s * np.eye(f_s)).dot(X.T.dot(y))
    
    def pred(self, X):
        return X.dot(self.theta)