"""
Genetic Algorithm that evolves CNN architectures and hyper-parameters.

Chromosome   a Genome (see cnn.py): mixed integer / categorical / real genes
Fitness      validation accuracy after a short training run (ties: lower validation loss)
Selection    tournament (size 3)
Crossover    uniform: every gene comes from either parent with probability 1/2
Mutation     per gene: integer and ordered genes step to a neighbouring value,
             kernel is reset at random, real genes get Gaussian noise
Elitism      the best individuals survive unchanged
Cache        a network already trained is never trained again (same genome -> same fitness)

Generation 0 is a random population; evolve_iter() yields after each generation so the
web demo can draw the curve live.
"""

from dataclasses import dataclass, field

import numpy as np

from cnn import (DENSE_CHOICES, DROPOUT_RANGE, FILTER_CHOICES, KERNEL_CHOICES, LR_EXP_RANGE, MAX_BLOCKS,
                 Genome)


@dataclass
class GAConfig:
    pop_size: int = 16
    generations: int = 12
    tournament: int = 3
    crossover_rate: float = 0.9
    mutation_rate: float = 0.2        # per gene
    elite: int = 2
    seed: int = 0


@dataclass
class Individual:
    genome: Genome
    acc: float
    loss: float
    params: int
    seconds: float

    @property
    def rank(self):
        return (self.acc, -self.loss)

    def to_dict(self):
        return {"genome": self.genome.to_dict(), "acc": self.acc, "loss": self.loss,
                "params": self.params, "seconds": round(self.seconds, 2), "describe": self.genome.describe()}


@dataclass
class Evaluator:
    """Wraps the (expensive) fitness function with a cache and records every unique evaluation."""
    fitness_fn: callable
    cache: dict = field(default_factory=dict)
    trace: list = field(default_factory=list)           # unique evaluations in order

    def __call__(self, g):
        k = g.key()
        if k not in self.cache:
            r = self.fitness_fn(g)
            ind = Individual(g, r["acc"], r["loss"], r["params"], r["seconds"])
            self.cache[k] = ind
            self.trace.append(ind)
        ind = self.cache[k]
        return Individual(g, ind.acc, ind.loss, ind.params, ind.seconds)


def random_genome(rng):
    return Genome(n_blocks=int(rng.integers(1, MAX_BLOCKS + 1)),
                  filters=tuple(int(rng.choice(FILTER_CHOICES)) for _ in range(MAX_BLOCKS)),
                  kernel=int(rng.choice(KERNEL_CHOICES)),
                  dense=int(rng.choice(DENSE_CHOICES)),
                  dropout=float(rng.uniform(*DROPOUT_RANGE)),
                  lr_exp=float(rng.uniform(*LR_EXP_RANGE))).canonical()


def _step(choices, value, rng):
    i = choices.index(value) + int(rng.choice([-1, 1]))
    return choices[min(max(i, 0), len(choices) - 1)]


def crossover(a, b, rng):
    pick = lambda x, y: x if rng.random() < 0.5 else y
    return Genome(n_blocks=pick(a.n_blocks, b.n_blocks),
                  filters=tuple(pick(x, y) for x, y in zip(a.filters, b.filters)),
                  kernel=pick(a.kernel, b.kernel), dense=pick(a.dense, b.dense),
                  dropout=pick(a.dropout, b.dropout), lr_exp=pick(a.lr_exp, b.lr_exp))


def mutate(g, rng, rate):
    hit = lambda: rng.random() < rate
    return Genome(
        n_blocks=g.n_blocks + int(rng.choice([-1, 1])) if hit() else g.n_blocks,
        filters=tuple(_step(FILTER_CHOICES, f, rng) if hit() else f for f in g.filters),
        kernel=int(rng.choice(KERNEL_CHOICES)) if hit() else g.kernel,
        dense=_step(DENSE_CHOICES, g.dense, rng) if hit() else g.dense,
        dropout=g.dropout + rng.normal(0, 0.1) if hit() else g.dropout,
        lr_exp=g.lr_exp + rng.normal(0, 0.3) if hit() else g.lr_exp,
    ).canonical()


def evolve_iter(evaluator, cfg=GAConfig()):
    """Yields (generation, population sorted best-first) for generation 0..cfg.generations."""
    rng = np.random.default_rng(cfg.seed)
    pop = [evaluator(random_genome(rng)) for _ in range(cfg.pop_size)]
    for gen in range(cfg.generations + 1):
        pop.sort(key=lambda i: i.rank, reverse=True)
        yield gen, pop
        if gen == cfg.generations:
            return

        def select():
            idx = rng.choice(len(pop), cfg.tournament, replace=False)
            return max((pop[i] for i in idx), key=lambda i: i.rank).genome

        new = pop[:cfg.elite]                                                # elitism
        seen = {i.genome.key() for i in new}
        while len(new) < cfg.pop_size:
            a, b = select(), select()
            child = crossover(a, b, rng) if rng.random() < cfg.crossover_rate else a
            child = mutate(child, rng, cfg.mutation_rate)
            for _ in range(10):                      # keep diversity: no twins inside one population
                if child.key() not in seen:
                    break
                child = mutate(child, rng, max(cfg.mutation_rate, 0.3))
            seen.add(child.key())
            new.append(evaluator(child))
        pop = new


def random_search(evaluator, budget, seed=0):
    """Baseline with the same compute: sample random genomes until `budget` unique networks are trained."""
    rng = np.random.default_rng(seed + 1000)
    while len(evaluator.trace) < budget:
        evaluator(random_genome(rng))
    return max(evaluator.trace, key=lambda i: i.rank)
