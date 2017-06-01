#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 16:01:57 2017
F optimizers
@author: stijndc
"""
import os
import pandas as pd
import time
import numpy as np
from sklearn.metrics import fbeta_score
import random



def find_thresholds(y_train, p_train, y_valid, p_valid):
    best_anokas = optimise_f2_thresholds(y_train, p_train)

    fixed_threshold = 0.2
    p_valid_binary = p_valid > fixed_threshold
    score_valid = fbeta_score(y_valid, p_valid_binary, beta=2, average='samples')
    print('Score on the validation set with fixed threshold {}: {}'.format(fixed_threshold, score_valid))

    p_valid_binary_best = p_valid > best_anokas
    score_valid_best = fbeta_score(y_valid, p_valid_binary_best, beta=2, average='samples')
    print('Score on the validation set with best threshold: {}'.format(score_valid_best))
    print('Optimized thresholds: {}'.format(best_anokas))
    return(best_anokas)



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


def truncate_probabilities(probabilities):
    """If an image is cloudy with high confidence, set all other probabilities to zero."""
    probabilities[probabilities[:, 6] > 0.8,:] = [0]*6 + [0.8] + [0]*10
    return(probabilities)

# Find best f2 score threshold


def truncate_weathertags(probabilities):
    """Set all but one probabilities of weather tags to zero. Retain the weather tag with the largest posterior probability."""
    for r in range(probabilities.shape[0]):
        i = np.argmax(probabilities[r,[5,6,10,11]])
        probabilities[r,[5,6,10,11]] = 0
        probabilities[r, i] = 1

    return(probabilities)

def find_f2score_threshold(p_valid, y_valid):
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


# Find best threshold for each label individually with sim annealing?


#def return_score(threshold_set, p_set, true_set):
    #"""Return fbeta score for a set of predicted probabilities with a set of thresholds.
    #One threshold per label"""
    #return(fbeta_score(true_set, p_set > threshold_set, beta=2, average='samples'))

#f_score = lambda t_set, p_set, l_set: 1 - fbeta_score(l_set, p_set > t_set, beta=2, average='samples')

#def find_f2score_threshold_simannealing(p_set, l_set, best=0.2):
#    """Find best set of individual thresholds on a set of data and true labels."""
    #x0 = [random.random() for i in range(17)]
#    x0 = [best + random.random()/10. for i in range(17)]
#    print(x0)
#    minimizer_kwargs = {'args': (p_set, l_set, )}
#    best_set = scipy.optimize.basinhopping(func=f_score,
#                                           x0=x0,
#                                           minimizer_kwargs=minimizer_kwargs,
#                                           niter=50,
#                                           disp=True, T=2.0, stepsize=0.1)
#
#    return(best_set)
