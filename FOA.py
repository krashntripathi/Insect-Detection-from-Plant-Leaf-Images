import numpy as np
import random
import time

def FOA(X, objective_function, lb, ub, max_iter):
    # Fossa Optimization Algorithm (FOA)

    fossas = X
    n_fossas, dim = fossas.shape
    best_fossa = fossas[0]
    best_fitness = objective_function(best_fossa)
    fitness = objective_function(fossas)
    Convergence_curve = np.zeros(max_iter)
    t = 0
    start_time = time.time()

    while t < max_iter:
        for i in range(n_fossas):
            fitness[i] = objective_function(fossas[i])
            if fitness[i] < best_fitness:
                best_fossa = fossas[i]
                best_fitness = fitness[i]

        # Exploration Phase (Attack Lemur)
        for i in range(n_fossas):
            r = np.random.rand(dim)
            selected_fossa = random.choice(fossas)
            fossas[i] = fossas[i] + r * (selected_fossa - fossas[i])
            fossas[i] = np.clip(fossas[i], lb[i], ub[i])

        # Exploitation Phase (Chase Lemur)
        for i in range(n_fossas):
            r = np.random.rand(dim)
            step_size = (ub[i] - lb[i]) / (t + 1)
            fossas[i] = fossas[i] + (1 - 2 * r) * step_size
            fossas[i] = np.clip(fossas[i], lb[i], ub[i])

        Convergence_curve[t] = best_fitness
        print(f"Iteration {t}: Best Fitness = {best_fitness}")
        t += 1

    end_time = time.time()
    total_time = end_time - start_time
    return best_fitness, Convergence_curve, best_fossa, total_time

