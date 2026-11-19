"""Verify the cached solver with a direct history recursion and a known-type oracle."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from game import Context, FULL, Solver, TYPES


def independent_value(context, horizon, memory, mode, regime, types):
    """Expand observation histories directly, without the solver's transition tables."""
    weights = np.array([context.prior[i] for i in types])
    weights /= weights.sum()
    if horizon == 0:
        return 0.
    choices = []
    for a in (0, 1):
        reward = (-context.mismatch_cost if a != mode else
                  (1. if a == regime else context.low_reward))
        b = int((1 - context.rate) * memory + context.rate * 8 * a + .5)
        future = 0.
        for next_g in (0, 1):
            probability = context.flip_rate if regime != next_g else 1 - context.flip_rate
            groups = [[], []]
            for i in types:
                lower, upper = TYPES[i]
                if next_g == 0:
                    lower, upper = 8 - upper, 8 - lower
                next_mode = int(b >= upper if mode == 0 else b > lower)
                groups[next_mode].append(i)
            for next_y, group in enumerate(groups):
                mass = sum(weight for weight, i in zip(weights, types) if i in group)
                if mass:
                    future += probability * mass * independent_value(
                        context, horizon - 1, b, next_y, next_g, tuple(group))
        choices.append(reward + context.discount * future)
    return max(choices)


def main():
    rng = np.random.default_rng(631)
    errors = []
    for _ in range(24):
        context = Context(5, rng.uniform(.94,.99), rng.uniform(.12,.6), rng.uniform(.04,.45),
                          rng.uniform(.015,.3), float(rng.choice((.25,.5,.75))),
                          tuple(rng.dirichlet(np.ones(5))))
        solver = Solver(context)
        for regime in (0, 1):
            cached = max(solver.values(5, 0, 0, regime, FULL))
            expanded = independent_value(context, 5, 0, 0, regime, tuple(range(5)))
            errors.append(abs(cached - expanded))
        solver.values.cache_clear()
    assert max(errors) < 1e-12
    # A public posterior cannot depend on the private type realization.
    for type_id, (lo, hi) in enumerate(TYPES):
        s = (lo + hi) / 16
        r = 1 - s
        switching = (hi - lo) / 16
        np.testing.assert_allclose((s - switching) / (r + s), lo / 8)
        np.testing.assert_allclose((s + switching) / (r + s), hi / 8)
        assert r > s > 0
    result = {"independent_history_enumerations": len(errors), "max_error": max(errors),
              "best_response_threshold_pairs_checked": len(TYPES)}
    path = Path(__file__).resolve().parent / "oracle_audit.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()
