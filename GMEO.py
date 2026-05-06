import numpy as np
import time

def GMEO(X, fname, xmin, xmax, Max_iter):
    # Groupers and Moray Eels Optimization (GMEO)
    Npop, Dim = X.shape
    n = Npop // 2
    Psearch = Max_iter // 3
    Penc = Max_iter // 3
    Pattack = Max_iter - (Psearch + Penc)

    #  Fitness Initialization
    fitness = fname(X)
    best_idx = np.argmax(fitness)
    gbest = X[best_idx].copy()
    gbest_fit = fitness[best_idx]
    fitness4 = []

    # 1. PRIMARY SEARCH (Zig-Zag Exploration)
    start_time = time.time()
    for it in range(Psearch):
        for i in range(n):
            hops = 3
            best_local = X[i].copy()
            best_local_fit = fname(best_local.reshape(1, -1))[0]

            for h in range(hops):
                new_pos = X[i].copy()
                for d in range(Dim):
                    if h % 2 == 0:
                        new_pos[d] = np.random.uniform(X[i, d], xmax[i, d])
                    else:
                        new_pos[d] = np.random.uniform(xmin[i, d], X[i, d])
                new_pos = np.clip(new_pos, xmin[i], xmax[i])
                fit = fname(new_pos.reshape(1, -1))[0]
                if fit > best_local_fit:
                    best_local = new_pos.copy()
                    best_local_fit = fit
            X[i] = best_local.copy()
        fitness = fname(X)
        best_idx = np.argmax(fitness)
        if fitness[best_idx] > gbest_fit:
            gbest_fit = fitness[best_idx]
            gbest = X[best_idx].copy()
        fitness4.append(gbest_fit)

    # 2. PAIR ASSOCIATION (Fitness-based)
    groupers = X[:n]
    eels = X[n:]
    fit_g = fname(groupers)
    fit_e = fname(eels)
    idx_g = np.argsort(-fit_g)
    idx_e = np.argsort(-fit_e)
    groupers = groupers[idx_g]
    eels = eels[idx_e]

    # 3. ENCIRCLING / EXTENDED SEARCH
    for it in range(Penc):
        for i in range(n):
            G = groupers[i]
            E = eels[i]

            # Prey estimation (between G and E)
            D = E - G
            dist = np.linalg.norm(D) + 1e-9
            L = np.random.rand() * dist
            prey = G + (L / dist) * D

            #  Grouper Spiral Motion
            hops = 3
            for h in range(hops):
                w = 1 - 2 * (h / max(1, hops))
                k = 0.5
                D1 = prey - G
                G = D1 * np.exp(k * w) * np.cos(2 * np.pi * w) + prey
                G = np.clip(G, xmin[0], xmax[0])

            #  Eel Sinusoidal Motion
            A = np.random.rand(Dim)
            E = E + A * np.sin(2 * np.pi * np.random.rand())
            E = np.clip(E, xmin[0], xmax[0])
            groupers[i] = G
            eels[i] = E

        X = np.vstack((groupers, eels))
        fitness = fname(X)
        best_idx = np.argmax(fitness)

        if fitness[best_idx] > gbest_fit:
            gbest_fit = fitness[best_idx]
            gbest = X[best_idx].copy()
        fitness4.append(gbest_fit)

    # 4. ATTACKING & CATCHING (Exploitation)
    for it in range(Pattack):
        for i in range(Npop):
            r = np.random.rand(Dim)
            X[i] = X[i] + r * (gbest - X[i])
            X[i] = np.clip(X[i], xmin[i], xmax[i])
        fitness = fname(X)
        best_idx = np.argmax(fitness)
        if fitness[best_idx] > gbest_fit:
            gbest_fit = fitness[best_idx]
            gbest = X[best_idx].copy()
        fitness4.append(gbest_fit)
    time4 = time.time() - start_time
    return gbest_fit, np.array(fitness4), gbest, time4


