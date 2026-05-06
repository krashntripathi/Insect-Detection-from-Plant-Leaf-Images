import numpy as np
import time


# Apiary Organizational-based Optimization Algorithm
def AOOA(population, objective_function, lb, ub, generations):
    theta = 0.5
    N, dim = population.shape
    fitness = objective_function(population)

    # Select Qi (Best Solution)
    best_fitness = np.min(fitness)
    best_idx = np.argmin(fitness)
    Qi = population[best_idx, :]
    new_b1 = Qi
    # Divide remaining solutions into workers (w) and drones (d)
    workers = np.copy(population)
    workers = np.delete(workers, best_idx, axis=0)  # Remove best solution from workers
    drones = np.copy(workers)

    # Initialize worker's age
    worker_age = np.ones(N - 1)

    # Convergence curve
    Convergence_curve = np.zeros(generations)

    # Main loop
    start_time = time.time()
    for t in range(generations):
        # Drone Exchange
        for i in range(N):
            for j in range(int(np.floor(N / 2))):
                # Select two hives (ha and hb) randomly
                ha = np.random.randint(0, N - 1)
                hb = np.random.randint(0, N - 1)

                # Select two drones (d1 from ha and d2 from hb) randomly
                d1 = drones[ha, :]
                d2 = drones[hb, :]

                # Exchange drones between hives
                drones[ha, :] = d2
                drones[hb, :] = d1

        # Fertilization and Bees Breeding
        for m in range(int(np.floor(N / 2))):
            # Select dbest and wbest in hive i
            dbest = np.min(drones, axis=0)
            wbest = np.min(workers, axis=0)

            r = np.random.rand(1, dim)

            # Fertilize Qi with dbest to generate new bees
            new_b1 = Qi + r * (dbest - Qi)
            # Fertilize Qi with wbest to generate new bees
            new_b2 = Qi + r * (wbest - Qi)

            # Apply Worker Lifecycle
            for k in range(N - 1):
                if worker_age[k] <= 10:
                    # Apply simple fertilization on worker
                    workers[k, :] = new_b1 + np.random.rand(1, dim) * (new_b2 - new_b1)
                elif worker_age[k] <= 25:
                    # Apply 2-optima fertilization on worker
                    workers[k, :] = new_b1 + 0.5 * np.random.rand(1, dim) * (new_b2 - new_b1)
                elif worker_age[k] <= 50:
                    # Apply 3-optima fertilization on worker
                    workers[k, :] = new_b1 + 0.33 * np.random.rand(1, dim) * (new_b2 - new_b1)
                elif worker_age[k] <= 60:
                    # Apply 2-optima fertilization on worker
                    workers[k, :] = new_b1 + 0.5 * np.random.rand(1, dim) * (new_b2 - new_b1)
                else:
                    # Drop worker
                    workers[k, :] = np.random.rand(1, dim) * (ub - lb) + ub
                worker_age[k] += 1

        # Queen Investiture
        new_fitness = objective_function(new_b1)
        if new_fitness - best_fitness >= theta:
            Qi = new_b1

        # Fading Out
        worst_idx = np.argmax(fitness)
        workers = np.delete(workers, worst_idx, axis=0)
        drones = np.delete(drones, worst_idx, axis=0)

        # Swarming
        h = 0
        if workers.shape[0] >= int(np.floor(N / 2)):
            # Split hive into two
            half_size = int(np.floor(workers.shape[0] / 2))
            workers = workers[:half_size, :]
            drones = drones[:half_size, :]

        # Update population and fitness
        population = np.vstack([workers, drones, Qi])
        fitness = objective_function(population)
        best_fitness = np.min(fitness)
        best_idx = np.argmin(fitness)
        Qi = population[best_idx, :]

        # Update convergence curve
        Convergence_curve[t] = best_fitness

    elapsed_time = time.time() - start_time
    best_solution = Qi
    return best_solution, Convergence_curve, best_fitness, elapsed_time