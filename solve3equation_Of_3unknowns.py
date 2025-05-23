import numpy as np
import random
from sympy import symbols, Eq, sympify, lambdify

x, y, z = symbols('x y z')

GENERATIONS = 10000
POPULATION_SIZE = 5000
MUTATION_RATE = 0.4
MUTATION_RANGE = 10000
PICK_NUMBER_FROM_PREV_GEN = 1000
PICK_NUMBER_FOR_PARENTS = 1000
NUMBER_OF_EQUATIONS = 3
STAGNATION_LIMIT = 30
MAX_VALUE = 1e7
MIN_VALUE = -1e7


def initial_generation():
    return np.random.uniform(MIN_VALUE, MAX_VALUE, size=(POPULATION_SIZE, 3))


def parse_equation_string(equation_string):
    left_expression, right_value = equation_string.split("=")
    left_expr = sympify(left_expression.strip())
    right_expr = float(right_value.strip())
    return Eq(left_expr, right_expr)


def create_functions(parsed_eqs):
    funcs = []
    for eq in parsed_eqs:
        f = lambdify((x, y, z), eq.lhs - eq.rhs, 'numpy')
        funcs.append(f)
    return funcs


def crossover(p1, p2):
    alpha = random.random()
    return alpha * p1 + (1 - alpha) * p2


def mutate(individual):
    if random.random() > MUTATION_RATE:
        return individual
    noise = np.random.normal(0, MUTATION_RANGE, size=3)
    mutated = np.clip(individual + noise, MIN_VALUE, MAX_VALUE)
    return mutated


def q2fitness(ind, funcs):
    x_val, y_val, z_val = ind
    total_error = 0
    try:
        for f in funcs:
            val = f(x_val, y_val, z_val)
            total_error += abs(val)
    except Exception:
        return float('inf')
    return total_error


def select_parent(population, scores):
    indices = random.sample(range(PICK_NUMBER_FOR_PARENTS), 20)
    best_index = min(indices, key=lambda i: scores[i])
    return population[best_index]


input_equations = []
for i in range(NUMBER_OF_EQUATIONS):
    equation = input(f"Enter equation {i+1} (e.g., 2*x + 3*y - z = 10): ")
    input_equations.append(equation)

parsed_eqs = [parse_equation_string(eq) for eq in input_equations]
funcs = create_functions(parsed_eqs)

population = initial_generation()
stagnation_count = 0
best_loss = float('inf')
previous_best = None

for generation in range(GENERATIONS):
    scores = np.array([q2fitness(index, funcs) for index in population])
    sorted_indices = np.argsort(scores)
    population = population[sorted_indices]
    scores = scores[sorted_indices]

    current_best = population[0]
    current_loss = scores[0]

    if current_loss <= 0.0000001:
        break

    if previous_best is not None and np.allclose(current_best, previous_best, rtol=1e-5, atol=1e-5):
        stagnation_count += 1
    else:
        stagnation_count = 0

    previous_best = current_best.copy()

    if stagnation_count >= STAGNATION_LIMIT:
        print(
            f"🛑 Stopped due to stagnation after {generation + 1} generations")
        break

    next_generation = population[:PICK_NUMBER_FROM_PREV_GEN].tolist()

    while len(next_generation) < POPULATION_SIZE:
        p1 = select_parent(population, scores)
        p2 = select_parent(population, scores)
        child = crossover(p1, p2)
        child = mutate(child)
        next_generation.append(child)

    population = np.array(next_generation)

    print(
        f"Gen {generation+1}: x={current_best[0]:.4f}, y={current_best[1]:.4f}, z={current_best[2]:.4f}, Loss={current_loss:.8f}")

answer = population[0]
final_loss = q2fitness(answer, funcs)

print("\n✅ Best solution found:")
print(f"x = {answer[0]:.2f}")
print(f"y = {answer[1]:.2f}")
print(f"z = {answer[2]:.2f}")
print(f"🔍 Total error: {final_loss:.2f}")
print(f"🧬 Stopped after {generation + 1} generations")
