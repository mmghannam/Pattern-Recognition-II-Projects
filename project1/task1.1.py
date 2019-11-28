from numpy import load, transpose, mean
from numpy.linalg import eig, eigh, svd, norm
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import normalize


def show_image(img_number, images_matrix, cmap='gray', size=(19, 19)):
    img_data = images_matrix[:, img_number].reshape(size)
    plt.imshow(img_data, cmap=cmap)
    plt.xticks([])
    plt.yticks([])
    plt.show()


def plot_eigen_values(eig, eigh, sigma):
    number_of_values = len(eig)
    plt.subplot(3, 1, 1)
    plt.title('eig eigne values of C')
    plt.bar(range(number_of_values), eig)

    plt.subplot(3, 1, 2)
    plt.title('eigh eigen values of C')
    plt.bar(range(number_of_values), eigh)

    plt.subplot(3, 1, 3)
    plt.title('squared singular values of X')
    plt.bar(range(number_of_values), map(lambda x: x ** 2, sigma))
    plt.tight_layout()

    plt.savefig("Task 1-1")
    # plt.show()


if __name__ == '__main__':
    # load data
    X = load('Data/faceMatrix.npy').astype('float')

    # show first 10 images
    # for i in range(10): show_image(i, X)

    # make column mean = 0
    X = normalize(X - mean(X, axis=0), axis=0)

    C = X.dot(transpose(X))

    # eig spectral decomposition
    eig_values, eig_vectors = eig(C)

    # eigh spectral decomposition
    eigh_values, eigh_vectors = eigh(C)

    # check mean square difference (result is that eigh values are close to eig values when they are flipped)
    # print(norm(eigh_values[::-1] - eig_values))

    # compute singular value decomposition
    u, s, vh = svd(X, full_matrices=True)

    # compare the results of eigh_values to squared of sigma values
    # result is flipped eigh values are nearly equal to sigma squared values (difference is probably due to
    # numerical instability).
    # print(norm(eigh_values[::-1] - map(lambda x: x ** 2, s)))

    plot_eigen_values(eig_values, eigh_values, s)