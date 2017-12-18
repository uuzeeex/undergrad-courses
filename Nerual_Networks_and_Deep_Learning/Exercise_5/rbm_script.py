# -*- coding: utf-8 -*-

from rbm import RBM
from tensorflow.examples.tutorials.mnist import input_data
from plot_digits import *
import numpy as np

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

rbm = RBM(nv=28 * 28, nh=500)

for epoch in range(100):
    batch_xs, batch_ys = mnist.train.next_batch(1000)
    cost = rbm.train_step(batch_xs, lr=0.1)
    print('Epoch %d, cost =' % epoch, cost)

print('    Visible                        Reonstructed')
for batch in range(4):
    test_x = mnist.test.images[batch * 5 : batch * 5 + 5]
    restore_test = rbm.reconstruct(test_x)
    print_out = []
    for i in range(5):
        print_out += [test_x[i], restore_test[i]]
    print_out = np.array(print_out)
    plot_digits(print_out)
