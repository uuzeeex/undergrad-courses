# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 16:52:11 2017

@author: lamwa
"""

import data_loader as dl
import data_standardizers as ds
import ridge_regression_model as model

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

data_loader = dl.DataProcessor('prostate.data.txt')

X_train, y_train, X_test, y_test = data_loader.get_data()

X_standardizer = ds.StandardScore()
X_standardizer.gen(X_train)

y_standardizer = ds.Demean()
y_standardizer.gen(y_train)

X_train_std = X_standardizer.standardize(X_train)
X_test_std = X_standardizer.standardize(X_test)

y_train_std = y_standardizer.standardize(y_train)

log_delta_sqrs = np.linspace(-2, 4, 20)

relative_error_train = []
relative_error_test = []
theta_path = np.empty([8, 0])

for log_d_s in log_delta_sqrs:
    ridge_regression = model.RidgeRegression(delta_sqr=np.power(10, log_d_s))
    ridge_regression.fit(X_train_std, y_train_std)
    theta_path = np.c_[theta_path, ridge_regression.theta]
    
    y_train_pred_std = ridge_regression.pred(X_train_std)
    y_train_pred = y_standardizer.destandardize(y_train_pred_std)
    relative_error_train.append(np.linalg.norm(y_train - y_train_pred) / np.linalg.norm(y_train))
    
    y_test_pred_std = ridge_regression.pred(X_test_std)
    y_test_pred = y_standardizer.destandardize(y_test_pred_std)
    relative_error_test.append(np.linalg.norm(y_test - y_test_pred) / np.linalg.norm(y_test))

colors = cm.rainbow(np.linspace(0, 1, 8))
labels = ['lcavol', 'lweight', 'age', 'lbph', 'svi', 'lcp', 'gleason', 'pgg45']
for i, c in zip(range(8), colors):
    plt.plot(log_delta_sqrs, theta_path[i].tolist(), color=c, label=labels[i])
plt.legend(loc='upper right', prop={'size': 10})
plt.xlabel(r'$\log_{10}\delta^2$')
plt.ylabel(r'$\theta$')
plt.savefig('plot2_1.pdf', format='pdf', dpi=1000)
plt.savefig('plot2_1.eps', format='eps', dpi=1000)

plt.close()

plt.plot(log_delta_sqrs, relative_error_train, 'r', label="train")
plt.plot(log_delta_sqrs, relative_error_test, 'b', label="test")

plt.legend(loc='upper left')
plt.xlabel(r'$\log_{10}\delta^2$')
plt.ylabel('relative error')
plt.savefig('plot2_2.eps', format='eps', dpi=1000)
