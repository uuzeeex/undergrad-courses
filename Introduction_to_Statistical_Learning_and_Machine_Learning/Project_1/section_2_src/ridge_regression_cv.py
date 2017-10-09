# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 13:28:29 2017

@author: lamwa
"""

import data_loader as dl
import data_standardizers as ds
import ridge_regression_model as model

import numpy as np
import matplotlib.pyplot as plt

data_loader = dl.DataProcessor('prostate.data.txt')

X_folds, y_folds = data_loader.get_folds()

log_delta_sqrs = np.linspace(-2, 4, 20)

relative_error_cv_avg = []
for log_d_s in log_delta_sqrs:

    ridge_regression = model.RidgeRegression(delta_sqr=np.power(10, log_d_s))
    relative_error_cv = []
    
    for i, X_fold, y_fold in zip(range(len(X_folds)), X_folds, y_folds):
        X_train = np.vstack(X_folds[: i] + X_folds[(i + 1) :])
        y_train = np.concatenate(y_folds[: i] + y_folds[(i + 1) :])
        X_test = X_fold
        y_test = y_fold
        
        X_standardizer = ds.StandardScore()
        X_standardizer.gen(X_train)

        y_standardizer = ds.Demean()
        y_standardizer.gen(y_train)
        
        X_train_std = X_standardizer.standardize(X_train)
        y_train_std = y_standardizer.standardize(y_train)
        
        ridge_regression.fit(X_train_std, y_train_std)
        
        X_test_std = X_standardizer.standardize(X_test)
        y_test_pred_std = ridge_regression.pred(X_test_std)
        y_test_pred = y_standardizer.destandardize(y_test_pred_std)
        relative_error_cv.append(np.linalg.norm(y_test - y_test_pred) / np.linalg.norm(y_test))
        
    relative_error_cv_avg.append(np.mean(relative_error_cv))

plt.plot(log_delta_sqrs, relative_error_cv_avg, 'r', label="cv")
plt.legend(loc='upper left')
plt.xlabel(r'$\log_{10}\delta^2$')
plt.ylabel('relative error (average)')
plt.savefig('plot2_3.eps', format='eps', dpi=1000)
