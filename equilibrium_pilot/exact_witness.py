"""Exact rational proof witness for loss from retaining only mean thresholds."""

import json
from fractions import Fraction as F
from functools import lru_cache
from pathlib import Path

TYPE_PAIRS = ((1, 3), (1, 5), (2, 4), (2, 5), (3, 4))
PRIORS = ((105, 295, 390, 10, 200), (295, 105, 10, 390, 200))
GAMMA, LOW, COST, HAZARD = F(97, 100), F(46, 100), F(45, 100), F(15, 100)
HORIZON = 8


def exact_root(weights):
    @lru_cache(None)
    def values(n, b, y, g, remaining_types):
        if n == 0:
            return F(0), F(0)
        total = sum(weights[i] for i in remaining_types)
        result = []
        for a in (0, 1):
            reward = -COST if a != y else (F(1) if a == g else LOW)
            next_b = (b + 8 * a + 1) // 2
            future = F(0)
            for next_g in (0, 1):
                context_probability = HAZARD if g != next_g else 1 - HAZARD
                next_groups = ([], [])
                for i in remaining_types:
                    lower, upper = TYPE_PAIRS[i]
                    if next_g == 0:
                        lower, upper = 8 - upper, 8 - lower
                    next_y = int(next_b >= upper if y == 0 else next_b > lower)
                    next_groups[next_y].append(i)
                for next_y, group in enumerate(next_groups):
                    if group:
                        probability = context_probability * F(sum(weights[i] for i in group), total)
                        future += probability * max(values(n - 1, next_b, next_y, next_g, tuple(group)))
            result.append(reward + GAMMA * future)
        return tuple(result)
    return values(HORIZON, 0, 0, 1, tuple(range(len(TYPE_PAIRS))))


def main():
    # Both full priors have the same two moments, exactly.
    moments = [tuple(sum(F(w, 1000) * pair[k] for w, pair in zip(weights, TYPE_PAIRS))
                     for k in (0, 1)) for weights in PRIORS]
    assert moments[0] == moments[1] == (F(9, 5), F(21, 5))
    roots = [exact_root(weights) for weights in PRIORS]
    assert roots[0][1] > roots[0][0]
    assert roots[1][0] > roots[1][1]
    scale = sum(GAMMA**t for t in range(HORIZON))
    a = (roots[0][1] - roots[0][0]) / scale
    b = (roots[1][0] - roots[1][1]) / scale
    regret_bound = a * b / (a + b)
    data = {
        "prior_integer_weights_over_1000": PRIORS, "mean_thresholds_in_grid_units": [1.8, 4.2],
        "horizon": HORIZON, "discount": .97, "low_coordination_reward": .46,
        "mismatch_cost": .45, "public_context_flip_probability": .15,
        "root_action": [1, 0],
        "normalized_action_gaps": [float(a), float(b)],
        "exact_gap_fractions": [str(a), str(b)],
        "minimax_regret_lower_bound": float(regret_bound),
        "exact_lower_bound_fraction": str(regret_bound),
    }
    target = Path(__file__).resolve().parent / "exact_witness.json"
    target.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps({k: v for k, v in data.items() if "fraction" not in k}, indent=2))


if __name__ == "__main__":
    main()
