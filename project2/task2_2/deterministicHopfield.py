from project2.task2_1.kRooksHopfield import kRooksHopfield
import numpy as np


def deterministic_behavior(Class):
    def asynchronous_choice(self, depth=2):
        original_state = self.state.copy()
        _, neuron = self.energy_search(original_state, self.depth)
        if self.local_minimum_check():
            self.max_iterations = self.iterations + 1
        return neuron

    def local_minimum_check(self):
        if np.array_equal(self.previous_states[-1], self.previous_states[-2]):
            return True
        return False

    def energy_search(self, state, depth):
        lowest_energy = np.Inf
        best_i = -1
        if depth != 0:
            for i in range(len(state)):
                new_state = state.copy()
                new_state[i] = 1 if self.weight_matrix[i] @ state >= self.thresholds[i] else -1
                if np.array_equal(state, new_state):
                    best_i = i if best_i == -1 else best_i
                    continue
                energy, _ = self.energy_search(new_state, depth - 1)
                if energy < lowest_energy:
                    lowest_energy = energy
                    best_i = i
        else:
            return -0.5 * state @ self.weight_matrix @ state + self.thresholds @ state, best_i

        return lowest_energy, best_i

    def run(self, depth=1, synchronous=False, convergence_params=[]):
        self.depth = depth
        old_run(self, synchronous, convergence_params)

    Class.asynchronous_choice = asynchronous_choice
    Class.energy_search = energy_search
    Class.local_minimum_check = local_minimum_check
    old_run = Class.run
    Class.run = run


def compare(k, n, criteria):
    deterministic = kRooksHopfield(k)
    non_deterministic = kRooksHopfield(k)
    non_deterministic.multiple_runs(n=n, convergence_params=[criteria])
    deterministic_behavior(kRooksHopfield)
    deterministic.run(convergence_params=[criteria])
    return deterministic, non_deterministic


if __name__ == '__main__':
    k = 5
    criteria = 100
    n = 30
    deterministic, non_deterministic = compare(k, n, criteria)
    # print(deterministic, non_deterministic)
    print("Deterministic iter: {}\n state:\n {}\n energy: {}, solved: {}\n".format(str(deterministic.iterations),
                                                                                   str(deterministic.reshaped_state()),
                                                                                   deterministic.previous_energies[-1],
                                                                                   deterministic.solved()))
    print("Deterministic iter: {}\n state:\n {}\n energy: {}, solved: {}\n".format(str(non_deterministic.iterations),
                                                                                   str(
                                                                                       non_deterministic.reshaped_state()),
                                                                                   non_deterministic.previous_energies[
                                                                                       -1],
                                                                                   non_deterministic.solved()))
