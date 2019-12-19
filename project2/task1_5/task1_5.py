#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 12:53:42 2019

@author: shahnawaz
"""

from project2.hopfield import Hopfield
import numpy as np
import timeit
from scipy import spatial
import matplotlib.pyplot as plt


class argmaxHopfield(Hopfield):
    def __init__(self, k, lda, X, m, seed=100):
        self.m = m
        self.k = k
        self.lda = lda
        self.X = X
        super().__init__(m, seed)

    def initialize_state(self):
        import numpy as np
        self.state = np.ones(self.m) * -1

        self.state[np.random.randint(self.m, size=self.k)] = 1

    def initialize_weight_matrix(self):
        # self.weight_matrix = -0.5 * self.lda(np.ones((self.m, self.m))  -((self.k+1)/self.k)*(self.X) - self.lda* np.eye(self.m))
        self.weight_matrix = 0.5 * self.X - (0.5 * self.lda) * (np.ones((self.m, self.m)) - np.eye(self.m))
        # print(self.weight_matrix)
        # exit()

    def initialize_thresholds(self):
        c = 2 * self.k - self.m
        ones = np.ones(self.number_of_neurons)
        # thres = -0.25 * (np.matmul(self.X, ones) + 2.0*self.lda*c*ones)
        thres = -0.5 * ones.T@self.X  - 0.5 * self.lda * c * ones

        self.thresholds = thres.T

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

def group_show(full_data, group_size=(5, 5), cmap='gray', size=(19, 19)):
    # plot multiple graysacale data together
    fig = plt.figure(figsize=group_size)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    for i in range(group_size[0] * group_size[1]):
        ax = fig.add_subplot(group_size[0], group_size[1], i + 1, xticks=[], yticks=[])
        ax.imshow(full_data[:, i].reshape(size), cmap=cmap)
    plt.show()


if __name__ == '__main__':
    X = np.load('faceMatrix.npy').astype('float')
    X = X[:, :100] / 255
    #    X = np.array([[8,0,3,4,1,5,9,7,6,2]])
    V = spatial.distance.pdist(X.T, 'sqeuclidean')

    D_original = spatial.distance.squareform(V)
    # D = np.tril(D_original, -1)
    D = D_original
    states = []
    K = [5, 10, 25]
    for k in K:
        hopfld = argmaxHopfield(k, 2000, D, np.shape(D)[0])
        hopfld.multiple_runs(n=10, convergence_params=[1000])
        st = hopfld._state()
        states.append(hopfld._state())
        print(hopfld.__str__())
        print((hopfld._state()>0).sum())
    i = 0
    for state in states:
        print(str(K[i]) + "images which are farthest")
        images = np.zeros((361, 0))
        for j in range(len(state)):
            if state[j] > 0:
                print(j)
                img = np.expand_dims(X[:, j], 1)
                images = np.concatenate((images, img), axis=1)
            pass
        group_show(images, (1, K[i]))
        i = i + 1
    i = 0
    for state in states:
        print(str(K[i]) + "images which are farthest")
        images = np.zeros((361, 0))
        for j in range(len(state)):
            if state[j] > 0:
                print(j)
                img = np.expand_dims(X[:, j], 1)
                img = img.reshape((19,19))
                fig, ax = plt.subplots()
                
                plt.imshow(img, cmap='gray')
                plt.show()
                #images = np.concatenate((images, img), axis=1)
            pass
        #group_show(images, (1, K[i]))
        i = i + 1

    print("Done")
