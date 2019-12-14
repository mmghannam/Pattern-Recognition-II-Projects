from project2.hopfield import Hopfield
import numpy as np


class kRooksHopfield(Hopfield):
    def __init__(self, k, seed=100):
        self.k = k
        super().__init__(k * k, seed)

    def initialize_weight_matrix(self):
        zeros = np.zeros([self.k, self.k])
        identity = np.eye(self.k)
        j = np.ones([self.k, self.k]) - identity

        weight_row = -2 * np.array(
            [np.concatenate((np.tile(zeros, i), j, np.tile(zeros, self.k - i - 1)), axis=1)
             for i in range(self.k)]).reshape(self.number_of_neurons, self.number_of_neurons)
        weight_columns = -2 * np.array(
            [np.concatenate((np.tile(identity, i), zeros, np.tile(identity, self.k - i - 1)), axis=1)
             for i in range(self.k)]).reshape(self.number_of_neurons, self.number_of_neurons)
        self.weight_matrix = weight_row + weight_columns

    def initialize_thresholds(self):
        c = 2 * self.k - self.number_of_neurons
        ones = np.ones(self.number_of_neurons)

        threshold_rows = -2 * c * ones
        threshold_columns = -2 * c * ones

        self.thresholds = threshold_rows + threshold_columns

    def run(self, synchronous=False, convergence_params=[]):
        for i in range(convergence_params[0]):
            self.update(synchronous)
            self.previous_energies.append(self.energy())
            self.previous_states.append(self.state)



if __name__ == '__main__':
    k = 3
    x = kRooksHopfield(k)
    x.run(convergence_params=[100])
    print(x)
    y = np.array([[-0., - 2., - 2., - 2., - 0., - 0., - 2., - 0., - 0.],
                  [-2., - 0., - 2., - 0., - 2., - 0., - 0., - 2., - 0.],
                  [-2., - 2., - 0., - 0., - 0., - 2., - 0., - 0., - 2.],
                  [-2., - 0., - 0., - 0., - 2., - 2., - 2., - 0., - 0.],
                  [-0., - 2., - 0., - 2., - 0., - 2., - 0., - 2., - 0.],
                  [-0., - 0., - 2., - 2., - 2., - 0., - 0., - 0., - 2.],
                  [-2., - 0., - 0., - 2., - 0., - 0., - 0., - 2., - 2.],
                  [-0., - 2., - 0., - 0., - 2., - 0., - 2., - 0., - 2.],
                  [-0., - 0., - 2., - 0., - 0., - 2., - 2., - 2., - 0.]])
    state = np.array([-1., - 1., - 1., - 1., - 1., - 1., - 1., - 1., - 1.])
    for i in range(9):
        print(y[i] @ state)
    exit()
