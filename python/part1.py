#!/usr/bin/env python3
# # Part 1 of Coursework

print("""
Welcome to part 1 of the coursework,

Here we will be covering genetic algorithims
but you already knew that.

Note, my comments are the string literals, I will print
them for clarity as to where I am in the document.
and so the output makes a little more sense.
""")

import random
import itertools
from collections import Counter
# ## Parameters that control the algorithm
#
# these are the knobs we can fiddle
# with to get different results and tune the algorithm. You can use the
# generation number in particular to go through each generation and see it
# improve.

population_size = 10
generations = 20
mutation_rate = 6

print("""
# ## Question 1. Correct creation of the population
#
# here we need to generate an individual,
# An individual is represented as a lsttor of ints,
# rand-int will generate numbers from $[0 ... n)$
# In this case our genome size is 32 so we take 32 random values like so.
""")

def generate_individual():
    return [random.randint(0, 1) for _ in range(32)]

print(f"{generate_individual()=}\n")


print("we can then generate a population by repeatedly calling `generate-individual`")
population = [generate_individual() for _ in range(population_size)]

print("\n".join(str(ind) for ind in population))

print("""
# ## Question 2 Correct use of a fitness function
#
# In this case our fitness function is looking for the amount of 1s in an
# individual, In this case to find them we can count up the amount of 1s and 0s
# in each individual like so.
""")

example_fitness = Counter(ind := generate_individual())
print(ind, dict(example_fitness), sep = " | ")

print("""
# In other other words the stronger individual has more amount of 1s
# So in other other other words the strongest individual has the higest sum, upto 32.
""")
def fitness(indiviual):
    return sum(indiviual)

print(f"{ind} | {fitness(ind)=}")

print("""
# to get the entire fitness of our population we map the fitness and sort decending
""")

print(sorted([fitness(individual) for individual in population], reverse=True))

print("""
# ## Question 3. Correct use of GA operations
# ### selection
#
# selection is the process of selecting individuals for the next generation.
# The form I will be using is a weighted random selection, otherwise known as
# roullete wheel selection. Our select function will calculate the cumulative
# fitness then will select $n$ amount of times where $n$ is the population size.
""")

def select(population, fitnesses):
    total_fitness = sum(fitnesses)
    normalized = [fitness / total_fitness for fitness in fitnesses]
    cumulative = list(itertools.accumulate(normalized))

    new_population = []
    population_size = len(population)

    for _ in range(population_size):
        rand_n = random.random()
        for curr, pop in zip(cumulative, population):
            if rand_n <= curr:
                new_population.append(pop)
                break

    return new_population

print("before:", *population, sep = "\n")
print("after :", *select(population, [fitness(ind) for ind in population]), sep = "\n")

print("""
# ### mutation
#
# Mutation is when we select random alliel and twiddle it.
""")

# this helper gets $n$ random indexes, we use a set to make sure each is unique.
# if the amount of indexes is less than the number needed we add one more
# to the set and check again.
def get_indexes(n):
    indexes = set()
    while len(indexes) < n:
        indexes.add(random.randint(0, 31))  # Generating indexes from 0 to 31
    return indexes

def mutate(mutation_value, individual):
    indexes = get_indexes(mutation_value)
    for index in indexes:
        individual[index] = 1
    return individual

print("""
# For this example I have chosen to use the dummy value 5,
# as the fifth letter of the alphabet is e, e standing for example.
""")

print(mutate(mutation_rate, [5] * 32))

print("""
# ### crossover
#
# crossover is when we take two parents and cut them at some kind of random
# interval, this gives us two new children.
""")
# this helper function takes a lst and splits it, returning the two halfs,
# this is an $O(1)$ operation which is the main reason we used lsttors
# in the first place
def lst_split(lst, cut_point):
    return lst[:cut_point], lst[cut_point:]

def crossover(individuals):
    ind1, ind2 = individuals
    cut_point = random.randint(0, 32)  # Cut point between 0 and 32 (inclusive)
    ind1_f, ind1_l = lst_split(ind1, cut_point)
    ind2_f, ind2_l = lst_split(ind2, cut_point)
    return [ind1_f + ind2_l, ind2_f + ind1_l]

print(*crossover([[1 for _ in range(32)], [0 for _ in range(32)]]), sep = "\n")
# ## Question 4. Demonstrably working GA

# ### Termination
# our termination conditions is when we either get an individual with 32 1s or we will reach `@generations`
def terminate(fitness, generation, generations):
    highest_fitness = sorted(fitness, reverse=True)[0]
    return highest_fitness == 32 or generation == generations

# This helper does one pass over the population, performing crossover and
# mutation on all of the individuals in the population.
def evolve(population):
    new_population = []

    # Group individuals into pairs for crossover
    for i in range(0, len(population), 2):
        # Apply crossover to pairs
        crossover_result = crossover([population[i], population[i + 1]])

        # Apply mutation to each individual in the crossover result
        mutated_crossover_result = [mutate(mutation_rate, individual) for individual in crossover_result]

        new_population.extend(mutated_crossover_result)

    return new_population

print("""
# ### Training
# the training function brings this all together, we loop until
we want to terminate, when we do, return the population, forged in
battle.
""")
def train(population):
    generation = 0
    while True:
        fitnesses = [fitness(individual) for individual in population]

        if terminate(fitnesses, generation, generations):
            return population  # Return the final population if termination condition is met

        # Select individuals, then evolve the population
        population = evolve(select(population, fitnesses))
        generation += 1

trainedpop = train(population)

print("before:",
      *sorted(
          zip(population, [fitness(ind) for ind in population]),
          key = lambda x: x[1],
          reverse = True
      ),
      sep = "\n"
)

print("after:",
      *sorted(
          zip(trainedpop, [fitness(ind) for ind in trainedpop]),
          key = lambda x: x[1],
          reverse = True
      ),
      sep = "\n"
)

print("# With that we are done.")
