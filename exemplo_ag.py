import numpy as np
import matplotlib.pyplot as plt
import time
from typing import Union, TypeVar

np.random.seed(8889)

ArrayLike = TypeVar("TypeVar", np.ndarray, list)

FLOAT_MIN = -5.0
FLOAT_MAX = 5.0
BITS = 8
TOTAL_STEPS = 2 ** BITS - 1


def rosenbrock(x: ArrayLike, y: ArrayLike):
    return ((1.0 - x) ** 2) + 100.0 * ((y - x ** 2) ** 2)

def generate_population(size: Union[tuple, int], lower_limit: float, upper_limit: float):
    return np.random.uniform(lower_limit, upper_limit, size=size)

def encode(value: float) -> str:
    scaled_value = int(((value - FLOAT_MIN) / (FLOAT_MAX - FLOAT_MIN)) * TOTAL_STEPS)
    
    return f'{scaled_value:0{BITS}b}'

def decode(binary_str: str) -> float:
    int_value = int(binary_str, 2)
    
    return FLOAT_MIN + (int_value / TOTAL_STEPS) * (FLOAT_MAX - FLOAT_MIN)

def crossing_over(parent1: ArrayLike, parent2: ArrayLike) -> ArrayLike:
    parent1_encoded = [encode(val) for val in parent1]
    parent2_encoded = [encode(val) for val in parent2]

    crossover_point = BITS // 2
    child1_encoded = [
        parent1_encoded[0][:crossover_point] + parent2_encoded[0][crossover_point:],
        parent1_encoded[1][:crossover_point] + parent2_encoded[1][crossover_point:],
    ]
    child2_encoded = [
        parent2_encoded[0][:crossover_point] + parent1_encoded[0][crossover_point:],
        parent2_encoded[1][:crossover_point] + parent1_encoded[1][crossover_point:],
    ]

    child1 = np.array([decode(gene) for gene in child1_encoded])
    child2 = np.array([decode(gene) for gene in child2_encoded])

    return child1, child2

def mutate(individual: ArrayLike, lower_limit: float, upper_limit: float) -> ArrayLike:
    return individual + np.random.uniform(lower_limit, upper_limit)

def main():
    iterations = 30
    best_iter = None
    best_fitness = None
    best_solution = None

    f_values = list()
    var_values = list()

    pm = 0.5

    population = generate_population((20, 2), FLOAT_MIN, FLOAT_MAX)
    for iter in range(iterations):
        fitness_scores = rosenbrock(population[:, 0], population[:, 1])

        parent1_idx, parent2_idx = fitness_scores.argsort()[:2]
        parent1 = population[parent1_idx]
        parent2 = population[parent2_idx]

        if best_iter is None:
            best_iter = iter
        if best_fitness is None:
            best_fitness = fitness_scores[parent1_idx]
            best_solution = parent1
            best_iter = iter

        if fitness_scores[parent1_idx] < best_fitness:
            best_fitness = fitness_scores[parent1_idx]
            best_solution = parent1
            best_iter = iter

            f_values.append(best_fitness)
            var_values.append(best_solution)

        child1, child2 = crossing_over(parent1, parent2)

        if np.random.uniform(0, 1) > pm:
            child2 = mutate(child2, -0.01, 0.01)
            child1 = mutate(child1, -0.01, 0.01)

        population = generate_population((28, 2), FLOAT_MIN, FLOAT_MAX)
        population = np.vstack([population, child1, child2])

    print(f"[INFO] Best Fitness       = {best_fitness:.8f}")
    print(f"[INFO] Found at iteration = {best_iter}")
    print(f"[INFO] Best solution      = {best_solution}")
    print("[INFO] Finished.")

    plt.plot(f_values)
    plt.grid(True)
    plt.xlabel("Iterações")
    plt.ylabel("Função Fitness")
    plt.savefig("funcao_fitness.png")
    plt.clf()

    var_values = np.asarray(var_values)
    plt.plot(var_values[:-1, 0], var_values[:-1, 1], "ro")
    plt.plot(var_values[-1, 0], var_values[-1, 1], "bo")
    plt.grid(True)
    plt.xlabel("")
    plt.ylabel("")
    plt.show()
    plt.savefig("var_values.png")


if __name__ == "__main__":
    t1 = time.perf_counter()
    main()
    t2 = time.perf_counter()
    spent = (t2 - t1) * 1000.0
    print(f"[INFO] Execution time = {spent:.2f} [ms]")
