#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 12:53:42 2019

@author: shahnawaz
"""

import functools

from project2.hopfield import Hopfield
import numpy as np
import timeit
from scipy import spatial

class argmaxHopfield(Hopfield):
    def __init__(self, k, lda, X, m, seed=None):
        self.m = m
        self.k = k
        self.lda = lda
        self.X = X
        super().__init__(m, seed)

    def initialize_weight_matrix(self):
        self.weight_matrix = -0.5 * self.lda * (np.ones((self.m, self.m))-np.eye(self.m))

    def initialize_thresholds(self):
        c = 2*self.k - self.m
        ones = np.ones(self.number_of_neurons)
        thres = -0.5 * (self.X + self.lda*c*ones)

        self.thresholds = thres

    def run(self, synchronous=False, convergence_params=[]):
        self.max_iterations = convergence_params[0]
        self.iterations = 0
        super().run(synchronous, convergence_params)

    def is_done(self, tolerance):
        self.iterations += 1
        return self.max_iterations == self.iterations

    def _state(self):
        return self.state

#    def solved(self):
#        reshaped_test = self.reshaped_state()
#        if np.all(np.sum(reshaped_test, 0) == 2 - self.k) and np.all(np.sum(reshaped_test, 1) == 2 - self.k):
#            return True
#        return False


if __name__ == '__main__':
    X = np.load('faceMatrix.npy').astype('float')
    X = X[:,:100]
    V = spatial.distance.pdist(X.T, 'sqeuclidean')
    D_original = spatial.distance.squareform(V)
    D = np.tril(D_original, -1)
    D = D.flatten() 
    
    np.random.seed(10)
#    D= np.random.randint(0, 50, (20))
    D = np.array([8,0,3,4,1,5,9,7,6,2])
    states = []
#    for k in [5, 10, 25]:
    for k in [1, 2, 3]:
        hopfld = argmaxHopfield(k, 10, D.T, len(D))
        hopfld.multiple_runs(n=1, convergence_params=[1])
        st = hopfld._state()
        states.append(hopfld._state())
        
    print("Done")
