"""Exact belief-space planning in a coordination game with private hysteresis."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

SIZE = 8
TYPES = np.array([(1, 3), (1, 5), (2, 4), (2, 5), (3, 4)])
FULL = (1 << len(TYPES)) - 1


@dataclass(frozen=True)
class Context:
    horizon: int
    discount: float
    low_reward: float
    mismatch_cost: float
    flip_rate: float
    rate: float
    prior: tuple


class Solver:
    def __init__(self, context):
        self.context = context
        self.records = {}
        self.known_q = None
        self.transition = {}
        for b in range(SIZE + 1):
            for y in (0, 1):
                for g in (0, 1):
                    for action in (0, 1):
                        next_b = int(np.floor((1 - context.rate) * b + context.rate * SIZE * action + .5))
                        for next_g in (0, 1):
                            lower = TYPES[:, 0] if next_g else SIZE - TYPES[:, 1]
                            upper = TYPES[:, 1] if next_g else SIZE - TYPES[:, 0]
                            outcomes = (next_b >= upper) if not y else (next_b > lower)
                            high = sum((1 << i) for i in np.flatnonzero(outcomes))
                            self.transition[b, y, g, action, next_g] = (next_b, high)
        self.weights = {mask: np.array([(mask >> i) & 1 for i in range(len(TYPES))]) *
                        np.asarray(context.prior) for mask in range(1, FULL + 1)}

    def reward(self, action, y, g):
        if action != y:
            return -self.context.mismatch_cost
        return 1. if action == g else self.context.low_reward

    def normalize(self, remaining):
        return sum(self.context.discount**np.arange(remaining))

    def belief(self, mask):
        weight = self.weights[mask]
        return weight / weight.sum()

    def children(self, b, y, g, mask, action):
        total = self.weights[mask].sum()
        for next_g in (0, 1):
            chance = self.context.flip_rate if g != next_g else 1 - self.context.flip_rate
            next_b, high = self.transition[b, y, g, action, next_g]
            for next_y in (0, 1):
                next_mask = mask & (high if next_y else FULL ^ high)
                if next_mask:
                    probability = chance * self.weights[next_mask].sum() / total
                    yield probability, next_b, next_y, next_g, next_mask

    @lru_cache(None)
    def values(self, remaining, b, y, g, mask):
        if remaining == 0:
            return (0., 0.)
        result = []
        for action in (0, 1):
            future = sum(prob * max(self.values(remaining - 1, nb, ny, ng, nm))
                         for prob, nb, ny, ng, nm in self.children(b, y, g, mask, action))
            result.append(self.reward(action, y, g) + self.context.discount * future)
        value = tuple(result)
        self.records[remaining, b, y, g, mask] = value
        return value

    def features(self, state):
        remaining, b, y, g, mask = state
        c = self.context
        belief = self.belief(mask)
        switches = []
        for action in (0, 1):
            for next_g in (0, 1):
                _, high = self.transition[b, y, g, action, next_g]
                switches.append(sum(belief[i] for i in range(len(TYPES)) if (high >> i) & 1))
        return np.r_[belief, remaining / 20, b / SIZE, y, g,
                     c.low_reward, c.mismatch_cost, c.flip_rate, c.rate, c.discount, switches]

    def prediction_features(self, state):
        remaining, b, y, g, mask = state
        c = self.context
        mean_lower, mean_upper = self.belief(mask) @ TYPES
        switches = []
        for action in (0, 1):
            for next_g in (0, 1):
                next_b, _ = self.transition[b, y, g, action, next_g]
                lower, upper = (mean_lower, mean_upper) if next_g else (SIZE - mean_upper, SIZE - mean_lower)
                switches.append(int(next_b >= upper if y == 0 else next_b > lower))
        return np.r_[mean_lower / SIZE, mean_upper / SIZE, remaining / 20, b / SIZE, y, g,
                     c.low_reward, c.mismatch_cost, c.flip_rate, c.rate, c.discount, switches]

    @lru_cache(None)
    def limited_values(self, depth, b, y, g, mask):
        if depth == 0:
            return (0., 0.)
        return tuple(self.reward(a, y, g) + self.context.discount * sum(
            prob * max(self.limited_values(depth - 1, nb, ny, ng, nm))
            for prob, nb, ny, ng, nm in self.children(b, y, g, mask, a)) for a in (0, 1))

    def population(self):
        return [dict(y=0, b=0, g=g, type_id=i) for g in (0, 1) for i in range(len(TYPES))]


def select_action(values):
    return int(values[1] > values[0])
