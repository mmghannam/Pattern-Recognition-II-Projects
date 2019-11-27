from numpy import load, transpose
from numpy.linalg import eig, eigh, svd, norm
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import timeit
import functools


if __name__ == '__main__':
    # load data
    X = load('faceMatrix.npy').astype('float')

    # normalize X as to make column mean 0
    C = X.dot(transpose(X))

    # measuring the timing of eig(C)
    ts = timeit.Timer(functools.partial(eig, C)).repeat(3, 100)
    print ("eig(C) took ", min(ts) / 100, " secs")

    # measuring the timing of eigh(C)
    ts = timeit.Timer(functools.partial(eigh, C)).repeat(3, 100)
    print ("eigh(C) took ", min(ts) / 100, " secs")

    # measuring the timing of eigh(C)
    ts = timeit.Timer(functools.partial(svd, X, full_matrices=True)).repeat(3, 100)
    print ("SVD function took ", min(ts) / 100, " secs")
