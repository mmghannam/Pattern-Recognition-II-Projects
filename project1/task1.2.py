from numpy import load, transpose
from numpy.linalg import eig, eigh, svd, norm
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import timeit
import functools


def eig_withC(X):
    C = X.dot(transpose(X))
    return eig(C)


def eigh_withC(X):
    C = X.dot(transpose(X))
    return eigh(C)


if __name__ == '__main__':
    # load data
    X = load('Data/faceMatrix.npy').astype('float')

    # measuring the timing of eig(C)
    ts = timeit.Timer(functools.partial(eig_withC, X)).repeat(3, 100)
    print ("eig(C) took ", min(ts) / 100, " secs")

    # measuring the timing of eigh(C)
    ts = timeit.Timer(functools.partial(eigh_withC, X)).repeat(3, 100)
    print ("eigh(C) took ", min(ts) / 100, " secs")

    # measuring the timing of eigh(C)
    ts = timeit.Timer(functools.partial(svd, X, full_matrices=True)).repeat(3, 100)
    print ("SVD function took ", min(ts) / 100, " secs")
