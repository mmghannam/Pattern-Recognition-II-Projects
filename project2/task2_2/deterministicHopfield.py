from project2.task2_1.kRooksHopfield import kRooksHopfield

import numpy as np


def asynchronous_choice(self):
    lowest_energy = np.Inf
    original_state = self.state.copy()
    for i in range(self.number_of_neurons):
        value = 1 if self.weight_matrix[i] @ self.state >= self.thresholds[i] else -1
        self.state[i] = value
        energy = self.energy()
        if energy < lowest_energy or (energy == lowest_energy and value != original_state[i]):
            lowest_energy = energy
            neuron = i
        self.state = original_state.copy()
    return neuron


def deterministic_behavior(Class):
    Class.asynchronous_choice = asynchronous_choice


def compare(k, n, criteria):
    deterministic = kRooksHopfield(k)
    nonDeterministic = kRooksHopfield(k)
    nonDeterministic.multiple_runs(n=n, convergence_params=[criteria])
    deterministic_behavior(kRooksHopfield)
    deterministic.run(convergence_params=[criteria])
    return deterministic, nonDeterministic


if __name__ == '__main__':
    k = 4
    criteria = 100
    n = 10
    x, y = compare(k, n, criteria)
    print(x, y)
    print("Deterministic state: {}, energy: {}\n".format(str(x.state), x.previous_energies[-1]))
    print("Non Deterministic state: {}, energy: {}\n".format(str(y.state), y.previous_energies[-1]))
