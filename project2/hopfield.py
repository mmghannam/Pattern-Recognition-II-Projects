import numpy as np


class Hopfield:
    def __init__(self, number_of_neurons, seed=100):
        np.random.seed(seed)
        self.number_of_neurons = number_of_neurons
        self.initialize()

    def initialize(self):
        self.initialize_state()
        self.initialize_thresholds()
        self.initialize_weight_matrix()
        self.initialize_previous_energies()

    reset = initialize  # alias for initialize method

    def initialize_previous_energies(self):
        self.previous_energies = [np.NINF, self.energy()]

    def initialize_state(self):
        self.state = self.__get_random_polar_state()

    def initialize_thresholds(self):
        self.thresholds = np.zeros(self.number_of_neurons)

    def initialize_weight_matrix(self):
        self.weight_matrix = np.ones(2 * [self.number_of_neurons])

    def update(self, synchronous=False):
        if synchronous:
            self.state = np.sign(self.weight_matrix @ self.state - self.thresholds)
            self.state[self.state == 0] = 1
        else:
            i = self.asynchronous_choice()
            self.state[i] = 1 if self.weight_matrix[i] @ self.state >= 0 else -1

    def asynchronous_choice(self):
        return np.random.randint(self.number_of_neurons)

    def run(self, synchronous=False, convergence_params=[]):
        while not self.is_done(*convergence_params):
            self.update(synchronous)
            self.previous_energies.append(self.energy())

    def energy(self):
        return -0.5 * self.state @ self.weight_matrix @ self.state + self.thresholds @ self.state

    def is_done(self, tolerance):
        return np.isclose(self.previous_energies[-1], self.previous_energies[-2], rtol=tolerance)

    def multiple_runs(self, n, synchronous=False, convergence_params=[]):
        lowest_energies = [np.Inf]
        for _ in range(n):
            self.run(synchronous, convergence_params)
            if self.previous_energies[-1] < lowest_energies[-1]:
                best_state = self.state.copy()
                lowest_energies = self.previous_energies.copy()
                self.reset()
        self.previous_energies = lowest_energies
        self.state = best_state

        return self

    def __get_random_polar_state(self):
        return 2 * np.random.randint(2, size=self.number_of_neurons) - 1


if __name__ == '__main__':
    hf = Hopfield(5)
    hf.multiple_runs(3, convergence_params=[0.1])
    expected_state = [1, -1, 1, 1, 1]
    assert all(x == y for (x, y) in zip(hf.state, expected_state)), "Behavior changed!"
