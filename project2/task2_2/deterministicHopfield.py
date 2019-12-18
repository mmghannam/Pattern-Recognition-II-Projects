from project2.task2_1.kRooksHopfield import kRooksHopfield
import numpy as np
import matplotlib.pyplot as plt


def deterministic_behavior(Class):
    def asynchronous_choice(self, depth=1):
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
        old_run(self, synchronous=synchronous, convergence_params=convergence_params)

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
                deterministic_behavior(kRooksHopfield)

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
