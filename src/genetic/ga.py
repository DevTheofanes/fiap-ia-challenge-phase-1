from __future__ import annotations

import json
import math
import random
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

from src.genetic.encoding import crossover, mutate, random_individual


@dataclass
class Individual:
    genes: Dict[str, Any]
    fitness: float | None = None
    params_resolved: Dict[str, Any] | None = None


@dataclass(frozen=True)
class GAConfig:
    pop_size: int
    n_generations: int
    mutation_rate: float
    crossover_rate: float
    tournament_k: int
    elite_pct: float
    seed: int


FitnessFn = Callable[[Dict[str, Any]], Tuple[float, Dict[str, Any]]]


def run_ga(
    model_name: str,
    fitness_fn: FitnessFn,
    config: GAConfig,
) -> Tuple[Individual, List[Dict[str, Any]]]:
    rng = random.Random(config.seed)
    population = _init_population(model_name, config.pop_size, rng)
    history: List[Dict[str, Any]] = []

    best_overall: Individual | None = None
    elite_size = max(1, int(config.pop_size * config.elite_pct))

    for generation in range(config.n_generations):
        gen_start = time.time()
        _evaluate_population(population, fitness_fn)

        population.sort(key=lambda ind: ind.fitness or -math.inf, reverse=True)
        best = population[0]
        best_overall = _pick_best(best_overall, best)

        fitness_values = [ind.fitness for ind in population if ind.fitness is not None]
        mean_fitness = float(sum(fitness_values) / max(1, len(fitness_values)))
        std_fitness = float(_stddev(fitness_values))

        history.append(
            {
                "generation": generation,
                "best_fitness": best.fitness,
                "mean_fitness": mean_fitness,
                "std_fitness": std_fitness,
                "best_genes": json.dumps(best.genes, sort_keys=True, default=str),
                "best_params": json.dumps(best.params_resolved or {}, sort_keys=True, default=str),
                "elapsed_sec": time.time() - gen_start,
            }
        )

        elites = [Individual(dict(ind.genes), ind.fitness, ind.params_resolved) for ind in population[:elite_size]]
        next_population = elites

        while len(next_population) < config.pop_size:
            parent_a = _tournament_select(population, config.tournament_k, rng)
            parent_b = _tournament_select(population, config.tournament_k, rng)

            if rng.random() < config.crossover_rate:
                child_genes = crossover(parent_a.genes, parent_b.genes, model_name, rng)
            else:
                child_genes = dict(parent_a.genes)

            child_genes = mutate(child_genes, model_name, rng, config.mutation_rate)
            next_population.append(Individual(child_genes))

        population = next_population

    if best_overall is None:
        raise RuntimeError("GA finished without evaluating any individuals.")
    return best_overall, history


def _init_population(model_name: str, pop_size: int, rng: random.Random) -> List[Individual]:
    return [Individual(random_individual(model_name, rng)) for _ in range(pop_size)]


def _evaluate_population(population: List[Individual], fitness_fn: FitnessFn) -> None:
    for individual in population:
        if individual.fitness is not None:
            continue
        fitness, params_resolved = fitness_fn(individual.genes)
        individual.fitness = fitness
        individual.params_resolved = params_resolved


def _tournament_select(population: List[Individual], k: int, rng: random.Random) -> Individual:
    k = max(1, min(k, len(population)))
    contenders = rng.sample(population, k)
    contenders.sort(key=lambda ind: ind.fitness or -math.inf, reverse=True)
    return contenders[0]


def _pick_best(current: Individual | None, candidate: Individual) -> Individual:
    if current is None:
        return Individual(dict(candidate.genes), candidate.fitness, candidate.params_resolved)
    if (candidate.fitness or -math.inf) > (current.fitness or -math.inf):
        return Individual(dict(candidate.genes), candidate.fitness, candidate.params_resolved)
    return current


def _stddev(values: List[float | None]) -> float:
    clean = [v for v in values if v is not None]
    if len(clean) <= 1:
        return 0.0
    mean = sum(clean) / len(clean)
    variance = sum((v - mean) ** 2 for v in clean) / (len(clean) - 1)
    return math.sqrt(variance)
