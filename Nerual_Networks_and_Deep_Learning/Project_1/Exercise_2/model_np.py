# -*- coding: utf-8 -*-

"""
Created on Thu Oct 19 15:19:58 2017

@author: lamwa
"""

import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

class Model():
    def __init__(self):
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
        self.n_in = 784  # 28 * 28
        self.n_out = 10  # 10 classes
        self.max_epochs = 10000  # max training steps 10000
        self.Weights = np.random.rand(self.n_in, self.n_out) # initialize W 0

        self.biases = np.zeros(self.n_out)  # initialize bias 0
        
        self.training_loss = []
        self.training_acc = []
        self.validation_loss = []
        self.validation_acc = []
        for i in range(self.max_epochs):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            batch_xs = np.array(batch_xs)
            batch_ys = np.array(batch_ys)
            
            self.training_loss.append(self.train(batch_xs, batch_ys, 0.0001))
            self.training_acc.append(self.compute_accuracy(batch_xs, batch_ys))
            if i % 500 == 0:
                accuracy_test = self.compute_accuracy(np.array(mnist.test.images[: 500]), np.array(mnist.test.labels[: 500]))
                self.validation_acc.append(accuracy_test)
                print("#" * 30)
                print("compute_accuracy:", accuracy_test)
                loss = self.cross_entropy(batch_ys, self.output(batch_xs))
                print("cross_entropy:", loss) # print out cross entropy loss function
                self.validation_loss.append(loss)
        
    def train(self, batch_x, batch_y, learning_rate):
        probs = self.output(batch_x)
        delta = probs - batch_y
        dW = batch_x.T.dot(delta)
        db = np.sum(delta, axis=0)
        self.Weights += -learning_rate * dW
        self.biases += -learning_rate * db
        return self.cross_entropy(batch_y, probs)
        
    def output(self, batch_x): # print out the predictions
        # avoiding overflow
        def softmax(x):
            e_x = np.exp(x - np.max(x))
            return e_x / (e_x.sum(axis=0)) + 1e-30  #
        prediction = np.add(np.dot(batch_x, self.Weights), self.biases)
        result = []
        for i in range(len(prediction)):
            result.append(softmax(prediction[i]))
        return np.array(result)
        
    def cross_entropy(self, batch_y, prediction_y): # cross entropy function
        cross_entropy = - np.mean(
            np.sum(batch_y * np.log(prediction_y), axis=1))
        return cross_entropy
        
    def compute_accuracy(self, xs, ys): # computing the accuracy
        pre_y = self.output(xs)
        pre_y_index = np.argmax(pre_y, axis=1)
        y_index = np.argmax(ys, axis=1)
        count_equal = np.equal(y_index, pre_y_index)
        count = np.sum([1 for e in count_equal if e ])
        sum_count = len(xs)
        return count * 1.0 / sum_count

model = Model()
epochs_t = np.linspace(0, 9999, 10000)
epochs_v = np.linspace(0, 9500, 20)
plt.plot(epochs_t, model.training_acc, 'r', label="training accuracy")
plt.plot(epochs_v, model.validation_acc, 'b', label="testing accuracy")
#plt.axis([0, 100, 0, 0.3])
plt.legend(loc='lower right')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.savefig('plot1.eps', format='eps', dpi=1000)
plt.show()

plt.close()

plt.plot(epochs_t, model.training_loss, 'r', label="training loss")
plt.plot(epochs_v, model.validation_loss, 'b', label="testing loss")
#plt.axis([0, 100, 0, 0.3])
plt.legend(loc='upper right')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.savefig('plot2.eps', format='eps', dpi=1000)
