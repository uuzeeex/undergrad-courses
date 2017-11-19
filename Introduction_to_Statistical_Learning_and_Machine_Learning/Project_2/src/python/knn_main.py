# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 19:41:49 2017

@author: lamwa
"""

import numpy as np
import utils
from run_knn import run_knn
import matplotlib.pyplot as plt

def compute_accuracy(valid_labels, valid_target): # computing the accuracy
    count_equal = np.equal(valid_labels, valid_target)
    count = np.sum([1 for e in count_equal if e])
    sum_count = len(count_equal)
    return count * 1.0 / sum_count

k_set = [1, 3, 5, 7, 9]
train_inputs, train_target = utils.load_train()
valid_inputs, valid_target = utils.load_valid()
test_inputs, test_target = utils.load_test()

valid_labels_set = []
for k in k_set:
    valid_labels_set.append(run_knn(k, train_inputs, train_target, valid_inputs))

valid_accuracy = []
for valid_labels in valid_labels_set:
    valid_accuracy.append(compute_accuracy(valid_labels, valid_target))

test_labels_set = []
for k in k_set:
    test_labels_set.append(run_knn(k, train_inputs, train_target, test_inputs))

test_accuracy = []
for test_labels in test_labels_set:
    test_accuracy.append(compute_accuracy(test_labels, test_target))

plt.plot(k_set, valid_accuracy, label="validation accuracy")
plt.plot(k_set, test_accuracy, label="test accuracy")
plt.axis([1, 9, 0.81, 1])
plt.legend(loc='upper right')
plt.xlabel('k')
plt.ylabel('accuracy')
plt.savefig('plot1.eps', format='eps', dpi=1000)
plt.show()