from project2.hopfield import Hopfield
from project2.task2_1.kRooksHopfield import kRooksHopfield

import numpy as np


def asynchronous_choice(self):
    lowest_energy = np.Inf
    original_state = self.state.copy()
    for i in range(self.number_of_neurons):
        value = 1 if self.weight_matrix[i] @ self.state >= self.thresholds[i] else -1
        self.state[i] = value
        energy = self.energy()
        if energy < lowest_energy or (energy <= value != original_state[i]):
            lowest_energy = energy
            neuron = i
        self.state = original_state.copy()
    return neuron


def deterministic_behavior(Class):
    Class.asynchronous_choice = asynchronous_choice


if __name__ == '__main__':
    k = 3
    x = kRooksHopfield(k)
    deterministic_behavior(kRooksHopfield)
    x.run(convergence_params=[100])
    print(x)
