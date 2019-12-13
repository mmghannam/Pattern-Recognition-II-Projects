import numpy as np


class Hopfield:
    def __init__(self, number_of_neurons, seed=100):
        np.random.seed(seed)
        self.number_of_neurons = number_of_neurons
        self.weight_matrix = self.initialize_weight_matrix()
        self.state = self.initialize_state()
        self.thresholds = self.initialize_thresholds()
        self.previous_energies = self.initialize_previous_energies()

    def initialize_previous_energies(self):
        return [np.NINF]

    def initialize_state(self):
        return 2 * np.random.randint(2, size=self.number_of_neurons) - 1

    def initialize_thresholds(self):
        return np.zeros(self.number_of_neurons)

    def initialize_weight_matrix(self):
        # raise NotImplementedError
        return np.ones(2 * [self.number_of_neurons])

    def update(self, synchronous=False):
        if synchronous:
            self.state = np.sign(self.weight_matrix @ self.state - self.thresholds)
            self.state[self.state == 0] = 1
        else:
            i = self.asynchronous_choice()
            self.state[i] = 1 if self.weight_matrix[i] @ self.state >= 0 else -1

    def asynchronous_choice(self):
        return np.random.randint(self.number_of_neurons)

    def run(self, criteria, synchronous=False):
        while not self.is_done(criteria):
            self.update(synchronous)

    def energy(self):
        return -0.5 * self.state @ self.weight_matrix @ self.state + self.thresholds @ self.state

    def is_done(self, criteria):
        self.previous_energies.append(self.energy())
        if np.isclose(self.previous_energies[-1], self.previous_energies[-2], rtol=criteria):
            return True
        return False

    def multiple_runs(self, n, criteria):
        lowest_energies = [np.Inf]
        for i in range(n):
            self.run(criteria)
            if self.previous_energies[-1] < lowest_energies[-1]:
                print('here')
                best_state = self.state.copy()
                print(best_state)
                lowest_energies = self.previous_energies.copy()
            self.initialize_state()
            self.initialize_previous_energies()
        self.state = best_state
        self.previous_energies = lowest_energies


# x = Hopfield(200)
# print(x.multiple_runs(3,0.1))
# print(x.state)
# print(x.weight_matrix)
# print(x.state)
# print(x.run(0.1))
# print(x.thresholds) [ 1 -1  1  1 -1 -1]
# print(x.number_of_neurons)
# x.update(synchronous=True)
# print(x.state)
