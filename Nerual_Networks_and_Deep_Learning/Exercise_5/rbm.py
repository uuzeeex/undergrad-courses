# -*- coding: utf-8 -*-

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class RBM(object):
    def __init__(self, nv=28*28, nh=500, \
                 weight=None, bias_h=None, bias_v=None, rns=None):
        
        self.nv = nv
        self.nh = nh
        
        # initialize parameters
        if rns is None:
            rns = np.random.RandomState(1234)
        
        if weight is None:
            weight = np.array(rns.uniform(low=-4 * np.sqrt(6. / (nh + nv)),
                              high=4 * np.sqrt(6. / (nh + nv)),
                              size=(nv, nh)))

        if bias_h is None:
            bias_h = np.zeros(nh)

        if bias_v is None:
            bias_v = np.zeros(nv)

        self.rns = rns
        self.weight = weight
        self.bias_h = bias_h
        self.bias_v = bias_v
    
    # propagate up
    def propup(self, vis):
        return sigmoid(np.dot(vis, self.weight) + self.bias_h)

    # propagate down
    def propdown(self, hid):
        return sigmoid(np.dot(hid, self.weight.T) + self.bias_v)
    
    # sample h given v
    def v2h(self, v0_sample):
        h1_mean = self.propup(v0_sample)
        h1_sample = self.rns.binomial(size=h1_mean.shape, n=1, p=h1_mean)
        return h1_mean, h1_sample

    # sample v given h
    def h2v(self, h0_sample):
        v1_mean = self.propdown(h0_sample)
        v1_sample = self.rns.binomial(size=v1_mean.shape, n=1, p=v1_mean)
        return v1_mean, v1_sample
    
    # gibbs sampling: h -> v -> h
    def h2v2h(self, h0_sample):
        v1_mean, v1_sample = self.h2v(h0_sample)
        h1_mean, h1_sample = self.v2h(v1_sample)
        return v1_mean, v1_sample, h1_mean, h1_sample

    # gibbs sampling: v -> h -> v
    def v2h2v(self, v0_sample):
        h1_mean, h1_sample = self.v2h(v0_sample)
        v1_mean, v1_sample = self.h2v(h1_sample)
        return v1_mean, v1_sample, h1_mean, h1_sample
    
    # training by taking several sequential states
    def train_step(self, dataset, lr=0.1):
        
        # 2 rounds of encoding and reconstruction
        pos_h_means, pos_h_samples = self.v2h(dataset)
        neg_v_means, neg_h_samples = self.h2v(pos_h_samples)
        nh_means = self.propup(neg_h_samples)
        
        batch_size = dataset.shape[0]

        # gradient descend
        self.weight += lr * (np.dot(dataset.T, pos_h_means) - np.dot(neg_h_samples.T, nh_means)) / batch_size
        self.bias_v += lr * np.mean(dataset - neg_h_samples, axis=0)
        self.bias_h += lr * np.mean(pos_h_means - nh_means, axis=0)
        
        # mean square as the observed cost
        cost = np.mean(np.square(dataset - neg_v_means))
        return cost
    
    # reconstruct by v -> h -> v
    def reconstruct(self, dataset):
        v_mean, _, _, _ = self.v2h2v(dataset)
        return v_mean