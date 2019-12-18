
# import numpy
import numpy as np
import scipy.spatial as spt

from scipy.sparse import csgraph

import sys
sys.path.append('../')


from hopfield import Hopfield

class ClusterHopfield(Hopfield):

    def __init__(self, data, number_of_neurons, seed=100):
        np.random.seed(seed)
        self.number_of_neurons = number_of_neurons
        self.data = data
        self.initialize()


    def initialize_weight_matrix(self):

        D = self.__compute_squared_EDM(self.data)
        mean = np.mean(D)
        A = self.__compute_adjancy_matrix(D, mean)

        L = csgraph.laplacian(A, normed=False)

        diag = L.diagonal()
        trans_diag = diag[np.newaxis].transpose()

        #self.weight_matrix = L + (2 / np.dot(self.__get_ones_transpose(self.data.shape[1]), diag)) * (diag * trans_diag)
        self.weight_matrix = 2 * A


    def __compute_squared_EDM(self, X):
        V = spt.distance.pdist(X.T, 'sqeuclidean')
        return spt.distance.squareform(V)

    def __compute_adjancy_matrix(self, D, mean):
        # determine dimensions of data matrix D
        m, n = D.shape

        A = np.zeros((n, n))
        # iterate over upper triangle of D
        for i in range(n):
            for j in range(n):
                if D[i, j] != 0 and D[i, j] <= mean:
                    A[i, j] = 1
                else:
                    A[i, j] = 0
        return A

    def __get_ones_transpose(self, n):
        return np.ones(n).transpose()
    
    """
    def energy(self):
        D = self.__compute_squared_EDM(self.data)
        mean = np.mean(D)
        A = self.__compute_adjancy_matrix(D, mean)

        L = csgraph.laplacian(A, normed=False)

        diag = L.diagonal()
        trans_diag = diag[np.newaxis].transpose()
        return -0.5 * self.state @ self.weight_matrix @ self.state + self.state @ ( L + A + (2 / np.dot(self.__get_ones_transpose(self.data.shape[1]), diag)) * (diag * trans_diag)) @ self.state
    """

def save_img(img_arr, save_path):

    img_arr = img_arr.reshape(19, 19)

    import cv2
    cv2.imwrite(save_path, img_arr)


if __name__=='__main__':
    X = np.load('faceMatrix.npy').astype('float')

    X = X[:, :100]

    hf = ClusterHopfield(X, X.shape[1])
    hf.run(convergence_params=[0.1])

    print("Cluster 1 : ", np.count_nonzero(hf.state == 1))
    print("Cluster 2 : ", np.count_nonzero(hf.state == -1))

    for i in range(X.shape[1]):
        img_arr = X[:,i]

        state = hf.state[i]
        save_img(img_arr, 'imgs/'+str(state)+'/'+str(i) + '.png')

