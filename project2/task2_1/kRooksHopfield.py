import functools

from project2.hopfield import Hopfield
import numpy as np
import timeit


class kRooksHopfield(Hopfield):
    def __init__(self, k, seed=100):
        self.k = k
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
        J_diagonal_matrix = np.array(J_diagonal_matrix).reshape(self.number_of_neurons, self.number_of_neurons)

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
        x = kRooksHopfield(k)
        x.multiple_runs(n=5, convergence_params=[100])
        print(x.reshaped_state())
