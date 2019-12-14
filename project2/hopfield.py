import numpy as np


class Hopfield:
    def __init__(self, number_of_neurons, seed=100, convergence_params=[]):
        np.random.seed(seed)
        self.number_of_neurons = number_of_neurons
        self.weight_matrix = self.initialize_weight_matrix()
        self.state = self.initialize_state()
        self.thresholds = self.initialize_thresholds()
        self.previous_energies = self.initialize_previous_energies()
        self.convergence_params = convergence_params

    def initialize_previous_energies(self):
        return [np.NINF]

    def initialize_state(self):
        return self.__get_random_polar_state()

    def initialize_thresholds(self):
        return np.zeros(self.number_of_neurons)

    def initialize_weight_matrix(self):
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

    def run(self, synchronous=False):
        while not self.is_done():
            self.update(synchronous)
            self.previous_energies.append(self.energy())

    def energy(self):
        return -0.5 * self.state @ self.weight_matrix @ self.state + self.thresholds @ self.state

    def is_done(self):
        tolerance = self.convergence_params[0]
        return self.previous_energies == [np.NINF] or \
               np.isclose(self.previous_energies[-1], self.previous_energies[-2], rtol=tolerance)

    def multiple_runs(self, n):
        lowest_energies = [np.Inf]
        for i in range(n):
            self.run()
            if self.previous_energies[-1] < lowest_energies[-1]:
                best_state = self.state.copy()
                print(best_state)
                lowest_energies = self.previous_energies.copy()
            self.initialize_state()
            self.initialize_previous_energies()
        self.state = best_state
        self.previous_energies = lowest_energies


    def __get_random_polar_state(self):
        return 2 * np.random.randint(2, size=self.number_of_neurons) - 1


if __name__ == '__main__':
    x = Hopfield(10, convergence_params=[0.1]) 
    x.multiple_runs(3)
