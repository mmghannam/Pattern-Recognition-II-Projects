import numpy as np
from numpy.linalg import norm
import scipy.linalg as la
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize


def get_distant_objects(X_tmp):
    X_tmp = X_tmp.transpose()
    x_i = X_tmp[np.random.choice(X_tmp.shape[0])] # choosing x_i randomly
    # finding three consecutive points with most distance from each other
    x_j = X_tmp[np.argmax(np.apply_along_axis(norm, 1, (X_tmp - x_i)))]
    x_k = X_tmp[np.argmax(np.apply_along_axis(norm, 1, (X_tmp - x_j)))]
    x_l = X_tmp[np.argmax(np.apply_along_axis(norm, 1, (X_tmp - x_k)))]
    X_tmp = X_tmp.transpose()
    return x_k, x_l

def approximate_major_component(X_k):
    # approximating major components
    x_a, x_b = get_distant_objects(X_k)
    return (x_a - x_b) / norm((x_a - x_b))




def group_show(full_data, group_size=(5,5) , cmap='gray', size=(19, 19)):
    # plot multiple graysacale data together
    fig = plt.figure(figsize=group_size) 
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    for i in range(group_size[0] * group_size[1]): 
        ax = fig.add_subplot(group_size[0], group_size[1], i+1, xticks=[], yticks=[]) 
        ax.imshow(full_data[:, i].reshape(size), cmap=cmap)
    plt.show()



if __name__ == '__main__':
    # load and show images
    X = np.load('Data/faceMatrix.npy').astype('float')
    group_show(X, group_size=(5,5))
    # normalize and apply SVD
    X = normalize(X - np.mean(X, axis=0), axis=0)
    u, s, vh = np.linalg.svd(X, full_matrices=True)

    # ploting left singular vectors
    group_show(u, group_size=(5,5))
    
    # initalize PCA vector
    v = np.zeros((X.shape[0], X.shape[0]))
    # fastmap iteration
    for i in range(25):
        v_k = approximate_major_component(X)
        v[i] = v_k
        identity = np.identity(X.shape[0])
        X = np.dot(identity - np.dot(v_k, v_k.transpose()), X)
    
    # plot fastmap PCA
    group_show(np.array(v).transpose(), group_size=(5,5))
