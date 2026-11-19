"""Descriptive local CPU timing on development histories, with per-query planner caches."""

import json
from statistics import median
from time import perf_counter

import numpy as np

from game import select_action
from run import FIT_SEEDS, HERE, action, arrays, contexts, feature_matrix, load_models, solve


def main():
    rng = np.random.default_rng(3951)
    model = load_models()["full", FIT_SEEDS[0]]
    records = []
    for context in contexts("development", 913561, 6):
        solver = solve(context)
        states = [s for s in solver.records if s[0] >= 8]
        selected = rng.choice(len(states), size=24, replace=False)
        records.extend((solver, states[i]) for i in selected)
    learned, planned = [], []
    for _ in range(5):
        start = perf_counter()
        for solver, state in records:
            x = feature_matrix(solver, [state], "full")
            int(arrays(model, x)[-1].item() > 0)
        learned.append((perf_counter() - start) / len(records))
        start = perf_counter()
        for solver, state in records:
            solver.limited_values.cache_clear()
            action(solver, "depth8", state)
        planned.append((perf_counter() - start) / len(records))
    result = {"queries_per_repeat": len(records), "repeat_count": 5,
              "learned_median_seconds": median(learned), "depth8_median_seconds": median(planned),
              "depth8_to_learned_latency_ratio": median(planned) / median(learned),
              "learned_parameters": sum(p.size for p in model),
              "interpretation": "Single-query CPU timing including feature computation, with a fresh lookahead cache per planning query. Exact filtering, model setup, training and teacher costs are excluded. Hardware timing is descriptive."}
    (HERE / "latency.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()
