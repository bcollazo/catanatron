# import random

# from deap import base
# from deap import creator
# from deap import tools

# creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# creator.create("Individual", list, fitness=creator.FitnessMax)  # type: ignore

# toolbox = base.Toolbox()
# # Attribute generator
# toolbox.register("attr_bool", random.randint, 0, 1)
# # Structure initializers
# toolbox.register(
#     "individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 100  # type: ignore
# )
# toolbox.register("population", tools.initRepeat, list, toolbox.individual)  # type: ignore


from deap import base, creator, tools  # deap utilities
import random
import numpy as np  # numerical computation
import matplotlib.pyplot as plt  # plotting

from catanatron_experimental.play import play_batch
from catanatron_experimental.machine_learning.players.value import ValueFunctionPlayer
from catanatron import Color
from catanatron_experimental.machine_learning.players.minimax import DEFAULT_WEIGHTS


creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)


IND_SIZE = 13  # chromosome length

toolbox = base.Toolbox()
toolbox.register("attribute", random.random)
toolbox.register(
    "individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=IND_SIZE
)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


def evaluate(ind):
    """Returns the fitness of an individual.
    This is your objective function."""
    # return (sum(ind),)  #  must return a tuple
    features = [
        # Where to place. Note winning is best at all costs
        "public_vps",
        "production",
        "enemy_production",
        "num_tiles",
        # Towards where to expand and when
        "reachable_production_0",
        "reachable_production_1",
        "buildable_nodes",
        "longest_road",
        # Hand, when to hold and when to use.
        "hand_synergy",
        "hand_resources",
        "discard_penalty",
        "hand_devs",
        "army_size",
    ]
    weights = {k: v * DEFAULT_WEIGHTS[k] for (k, v) in zip(features, ind)}
    # Have Color.BLUE be the experimental one
    players = [
        # AlphaBetaPlayer(Color.RED, 2, True),
        # AlphaBetaPlayer(Color.BLUE, 2, True, "C", weights),
        ValueFunctionPlayer(Color.RED, "C", params=DEFAULT_WEIGHTS),
        ValueFunctionPlayer(Color.BLUE, "C", params=weights),
    ]

    wins, results_by_player, _ = play_batch(200, players, quiet=True)
    vps = results_by_player[Color.BLUE]
    avg_vps = sum(vps) / len(vps)
    objective = 1000 * wins.get(Color.BLUE, 0) + avg_vps
    # breakpoint()
    return (objective,)


# Operators
toolbox.register("cross", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)


def generationalGA():
    pop = toolbox.population(n=50)  # Registered as a list
    CXPB = 0.5  # Crossover probability
    MUTPB = 0.2  # Mutation probability
    NGEN = 40  # Number of generations

    # Evaluate all population first
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Generate offspring
    for __ in range(NGEN):
        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))  # Generate a deep copy

        # Apply crossover
        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.cross(c1, c2)
                # Reset their fitness values
                del c1.fitness.values
                del c2.fitness.values

        # Mutate those selected
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                # Reset fitness values
                del mutant.fitness.values

        # Evaluate non-evaluated individuals in offspring
        invalid_inds = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_inds)
        for ind, fit in zip(invalid_inds, fitnesses):
            ind.fitness.values = fit

        # Replace entire population by the offspring
        pop[:] = offspring

    return pop


results = generationalGA()
fitnesses = [i.fitness.getValues()[0] for i in results]

bestpos = fitnesses.index(min(fitnesses))
best_individual = results[bestpos]
best_fitness = results[bestpos].fitness.getValues()[0]
print("Individual: {0}\n\nFitness: {1}".format(best_individual, best_fitness))
