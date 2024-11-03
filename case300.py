import numpy as np
import matplotlib.pyplot as plt
from pandapower.networks import case300
import tqdm

import functional as F


def main():
    sep300 = case300()

    init = F.initialize_pes(sep300)

    optimization_iters = 250
    iterations = 50
    num_individuals = 1000
    fitness = np.zeros((iterations, optimization_iters))
    perdas = np.zeros((iterations, optimization_iters))
    penaltygen = np.zeros((iterations, optimization_iters))
    penaltyv = np.zeros((iterations, optimization_iters))
    tempos = np.zeros((iterations, optimization_iters))
    solutions = np.zeros((iterations, 166))

    j, perdas_, pen_v, pen_gq, pen_tap, pen_bsh, global_best, tempo = F.optimize_ga(
        sep300, pt=0.7, rgap=0, zeta=0.01, psi=0.05, sigma=0.01, omega=0.01, max_iter=optimization_iters, num_individuals=num_individuals, c1=1, c2=0, v_amp=0.1, valor_inicial=0, step=0.01, wmax=0.9, relatorio=False, inicial=False)

    solutions[i, :] = global_best
    fitness[i, :] = j
    perdas[i, :] = perdas_
    penaltygen[i, :] = pen_gq
    penaltyv[i, :] = pen_v
    tempos[i, :] = tempo



if __name__ == "__main__":
    main()
