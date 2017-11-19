""" Methods for doing logistic regression."""

import numpy as np
from utils import sigmoid

def logistic_predict(weights, data):
    """
    Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to the bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
    Outputs:
        y:          :N x 1 vector of probabilities. This is the output of the classifier.
    """

    # TODO: Finish this function
    N, M = data.shape
    data_with_bias = np.c_[data, np.ones(N)]
    y = sigmoid(data_with_bias.dot(weights))

    return y

def evaluate(targets, y):
    """
    Compute evaluation metrics.
    Inputs:
        targets : N x 1 vector of binary targets. Values should be either 0 or 1
        y       : N x 1 vector of probabilities.
    Outputs:
        ce           : (scalar) Cross entropy.  CE(p, q) = E_p[-log q].  Here
                       we want to compute CE(targets, y).
        frac_correct : (scalar) Fraction of inputs classified correctly.
    """
    
    # TODO: Finish this function
    ce = -np.mean(np.sum(targets * np.log(y), axis=1))
    y_pred = np.array([[np.argmax([1. - prob, prob])] for prob in y])
    count_equal = np.equal(y_pred, targets)
    count = np.sum([1 for e in count_equal if e])
    sum_count = len(count_equal)
    frac_correct = count * 1.0 / sum_count
    return ce, frac_correct

def logistic(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
        targets:    N x 1 vector of binary targets. Values should be either 0 or 1.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:       The sum of the loss over all data points. This is the objective that we want to minimize.
        df:      (M+1) x 1 vector of derivative of f w.r.t. weights.
        y:       N x 1 vector of probabilities.
    """

    # TODO: Finish this function
    N, M = data.shape
    data_with_bias = np.c_[data, np.ones(N)]
    lin_res = data_with_bias.dot(weights)
    f = targets.T.dot(np.log(1. + np.exp(-lin_res))) + \
        (1 - targets).T.dot(np.log(1. + np.exp(lin_res)))
    y = sigmoid(lin_res)
    df = np.sum((targets * (y - 1.) + (1 - targets) * y) * \
                data_with_bias, axis=0).reshape((M + 1, 1))
    
    return f, df, y


def logistic_pen(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and 
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds 
                    to one data point.
        targets:    N x 1 vector of binary targets. Values should be either 0 or 1.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:             The sum of the loss over all data points. This is the objective that we want to minimize.
        df:            (M+1) x 1 vector of derivative of f w.r.t. weights.
    """

    # TODO: Finish this function
    N, M = data.shape
    f_, df_, y = logistic(weights, data, targets, hyperparameters)
    lmd = hyperparameters['weight_regularization']
    f = f_ + (lmd / 2.) * weights[: M].T.dot(weights[: M])
    df = df_ + np.vstack([lmd * weights[: M], [0.]])

    return f, df, y
