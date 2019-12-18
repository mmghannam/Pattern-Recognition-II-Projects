from project2.hopfield import Hopfield
import numpy as np
import matplotlib.pyplot as plt


class kRooksHopfield(Hopfield):
    def __init__(self, k, seed=100):
        self.k = k
        self.best_iterations = np.Inf
        super().__init__(k * k, seed)

    def initialize_weight_matrix(self):
        zeros = np.zeros([self.k, self.k])
        identity = np.eye(self.k)
        J = np.ones([self.k, self.k]) - identity

        weight_row = self.compute_weight_rows(J, zeros)

        weight_columns = self.compute_weight_columns(identity, zeros)

        self.weight_matrix = weight_row + weight_columns

    def compute_weight_rows(self, J, zeros):
        J_diagonal_matrix = []
        for diagonal_index in range(self.k):
            zeros_before_J = np.tile(zeros, diagonal_index)
            zeros_after_J = np.tile(zeros, self.k - diagonal_index - 1)
            column = np.concatenate((zeros_before_J, J, zeros_after_J), axis=1)
            J_diagonal_matrix.append(column)
        J_diagonal_matrix = np.array(J_diagonal_matrix).reshape(self.number_of_neurons,
                                                                self.number_of_neurons)

        return -2 * J_diagonal_matrix

    def compute_weight_columns(self, identity, zeros):
        complement_identity_matrix = []
        for diagonal_index in range(self.k):
            identities_before_zero = np.tile(identity, diagonal_index)
            identities_after_zero = np.tile(identity, self.k - diagonal_index - 1)
            column = np.concatenate((identities_before_zero, zeros, identities_after_zero), axis=1)
            complement_identity_matrix.append(column)
        complement_identity_matrix = np.array(complement_identity_matrix).reshape(self.number_of_neurons,
                                                                                  self.number_of_neurons)

        return -2 * complement_identity_matrix

    def initialize_thresholds(self):
        c = 2 - self.k
        ones = np.ones(self.number_of_neurons)

        threshold_rows = -2 * c * ones
        threshold_columns = -2 * c * ones

        self.thresholds = threshold_rows + threshold_columns

    def run(self, synchronous=False, convergence_params=[]):
        self.max_iterations = convergence_params[0]
        self.iterations = 0
        super().run(synchronous, convergence_params)
        if self.iterations < self.best_iterations:
            self.best_iterations = self.iterations

    def is_done(self, tolerance):
        self.iterations += 1
        return self.max_iterations == self.iterations or self.solved()

    def reshaped_state(self):
        return self.state.reshape((self.k, self.k))

    def solved(self):
        reshaped_test = self.reshaped_state()
        if np.all(np.sum(reshaped_test, 0) == 2 - self.k) and np.all(np.sum(reshaped_test, 1) == 2 - self.k):
            return True
        return False


if __name__ == '__main__':
    for k in [4, 8]:
        ns = [1, 10, 100]
        criterias = [10, 100, 1000]
        fig, axes = plt.subplots(nrows=3, ncols=3)
        fig.suptitle("K-Rooks K: " + str(k), fontsize=14)
        fig.subplots_adjust(wspace=1.0, hspace=1.0)

        i = -1
        for n in ns:

            for criteria in criterias:
                x = kRooksHopfield(k)

                x.multiple_runs(n=n, convergence_params=[criteria])
                print("K: {}\n iter: {}\n state:\n {}\n energy: {}, solved: {}\n".format(str(k), str(x.best_iterations),
                                                                                         str(x.reshaped_state()),
                                                                                         x.previous_energies[-1],
                                                                                         x.solved()))

                i += 1
                col = i % 3
                row = int(i / 3)
                img = x.reshaped_state()
                title = "R:" + str(n) + " I: " + str(x.best_iterations) + "/" + str(criteria) + ' G: ' + (
                    'T' if x.solved() else 'F')
                axes[row, col].matshow(img, cmap='gray')
                axes[row, col].set_title(title, pad=20)

        fig.subplots_adjust(top=0.85)
        fig.set_size_inches(6, 6)
        savtitle = "KRooks with K:" + str(k)
        plt.savefig(savtitle + ".jpg",
                    format='jpeg',
                    transparent=False,
                    bbox_inches='tight', pad_inches=0.01)
        plt.close()
