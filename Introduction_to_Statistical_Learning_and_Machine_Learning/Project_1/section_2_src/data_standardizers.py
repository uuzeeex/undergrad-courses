# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 17:18:23 2017

@author: lamwa
"""

import numpy as np

class DiscriminativeStandardizer(object):
    def gen(self, data):
        pass
    def standardize(self, data):
        pass
    def destandardize(self, data):
        pass

class StandardScore(DiscriminativeStandardizer):
    def gen(self, data):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
    
    def standardize(self, data):
        data_std = (data - self.mean) / self.std
        return data_std
    
    def destandardize(self, data_std):
        data = data_std * self.std + self.mean
        return data

class Demean(DiscriminativeStandardizer):
    def gen(self, data):
        self.mean = np.mean(data, axis=0)
    
    def standardize(self, data):
        data_std = data - self.mean
        return data_std
    
    def destandardize(self, data_std):
        data = data_std + self.mean
        return data