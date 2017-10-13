#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 16:01:57 2017
F optimizers by threshold adjustment
@author: stijndc
"""
import os
import pandas as pd
import time
import numpy as np
from sklearn.metrics import fbeta_score
import random


def find_thresholds(y_train, p_train, y_valid, p_valid):
    fixed_threshold = 0.2

    score_train = fbeta_score(y_train, p_train > fixed_threshold, beta=2, average='samples')
    score_valid = fbeta_score(y_valid, p_valid > fixed_threshold, beta=2, average='samples')

    print('Score on the training data with fixed threshold {}: {}'.format(fixed_threshold, score_train))
    print('Score on the validation data with fixed threshold {}: {}'.format(fixed_threshold, score_valid))

    # Look for optimal threshold on the training data
    optimized_threshold = optimise_f2_thresholds(y_train, p_train)

    score_train_best = fbeta_score(y_train, p_train > optimized_threshold, beta=2, average='samples')
    score_valid_best = fbeta_score(y_valid, p_valid > optimized_threshold, beta=2, average='samples')

    print('Score on the training data with best threshold: {}'.format(score_train_best))
    print('Score on the validation data with best threshold: {}'.format(score_valid_best))
    print('Optimized thresholds: {}'.format(optimized_threshold))

    return(optimized_threshold)

# --- BAC --- #

from sklearn.metrics import fbeta_score
import numpy as np
from scipy.optimize import minimize
import math


def f2_score(y_true, y_pred):
    y_true, y_pred, = np.array(y_true), np.array(y_pred)
    return fbeta_score(y_true, y_pred, beta=2, average='samples')


def cast_thresold(p, x):
    p2 = np.zeros_like(p)
    for i in range(17):
        p2[:, i] = (p[:, i] > x[i]).astype(np.int)
    return p2

def sigmoid(z, beta):
    fval = 1.0/(1+np.exp(-beta*z))
    g = np.array([beta*v*(1-v) for v in fval])
    return fval, g

def smooth_F_score(label, prob, z):
    beta, up, down = 1000, 0, 0
    pred, g = sigmoid(prob-z, beta)
    for i in range(len(label)):
        up += 5*label[i]*pred[i]
    for i in range(len(label)):
        down += 4*label[i] + pred[i]
    g = [-(5*down*label[i]*g[i] - up*g[i])/(down*down) for i in range(len(label))]
    return  up/down, g

def smooth_F2_score(x, labels, probs):
    score = 0
    grad = np.zeros(len(x))
    for i in range(labels.shape[0]):
        s, g = smooth_F_score(labels[i,:], probs[i,:], x)
        score += s
        grad += g
    grad = -grad/labels.shape[0]
    score = -score / labels.shape[0]
    return score, grad

def obj_func(x, labels, probs):
    s, g = smooth_F2_score(x, labels, probs)
    return s
def grad_func(x, labels, probs):
    s, g = smooth_F2_score(x, labels, probs)
    return g

def optimize_BAC(y, p, num_tries=100):
    best_score = -1
    best_x = np.ones(y.shape[1])*0.2
    for n_try in range(num_tries):
        print('Try {} of {}...'.format(n_try, num_tries))
        #x0 = np.random.rand(p.shape[1],1)
        x0 = np.random.beta(a=2,b=5,size=17) # Initialize x0 from beta distribution
        res = minimize(obj_func, x0, args=(y, p), method='BFGS', jac=grad_func, options = {'gtol': 1e-6, 'disp': False})
        score = f2_score(y, np.array(p > res.x, dtype='uint8'))
        if score > best_score:
            best_score = score
            best_x = res.x
    print(best_x)
    return best_x, best_score





















def optimise_f2_thresholds(y, p, verbose=False, resolution=100):
    """Optimize individual thresholds one by one. Code from anokas.
    Inputs
    ------
    y: numpy array, true labels
    p: numpy array, predicted labels
    """
    n_labels = y.shape[1]

    def mf(x):
        p2 = np.zeros_like(p)
        for i in range(n_labels):
            p2[:, i] = (p[:, i] > x[i]).astype(np.int)
        score = fbeta_score(y, p2, beta=2, average='samples')
        return score

    x = [0.2]*n_labels

    for i in range(n_labels):
        best_i2 = 0
        best_score = 0
        for i2 in range(resolution):
            i2 /= resolution
            x[i] = i2
            score = mf(x)
            if score > best_score:
                best_i2 = i2
                best_score = score
            x[i] = best_i2
            if verbose:
                print(i, best_i2, best_score)
    return x


def find_f2score_threshold(y_valid, p_valid):
    p_valid, y_valid = np.array(p_valid), np.array(y_valid)

    best = 0
    best_score = -1
    totry = np.arange(0,1,0.005)
    for t in totry:
        score = fbeta_score(y_valid, p_valid > t, beta=2, average='samples')
        if score > best_score:
            best_score = score
            best = t

    print('Best score: ', round(best_score, 5), ' @ threshold =', best)

    return best



def truncate_weathertags(probabilities):
    """Set all but one probabilities of weather tags to zero. Retain the weather tag with the largest posterior probability."""
    for r in range(probabilities.shape[0]):
        i = np.argmax(probabilities[r,[5,6,10,11]])
        probabilities[r,[5,6,10,11]] = 0
        probabilities[r, i] = 1

    return(probabilities)



def find_f2score_threshold_2D(p_0, p_1, y_valid):

    best_score = -1
    totry_0 = np.arange(0,1,0.05)# np.unique(p_0)
    totry_1 = np.arange(0,1,0.05)#np.unique(p_1)

    for t_0 in totry_0:
        for t_1 in totry_1:
            score = fbeta_score(y_valid, np.concatenate((p_0>t_0, p_1>t_1), axis=1), beta=2, average='samples')
            if score > best_score:
                best_score = score
                best_0 = t_0
                best_1 = t_1

    print('Best score: ', round(best_score, 5), '@ thresholds = ', (best_0, best_1))
    return((best_0,best_1))
