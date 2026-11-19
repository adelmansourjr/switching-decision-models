"""Fit a small value policy, freeze it, and evaluate exact expected held-out returns."""

import argparse
import hashlib
import json
import secrets
from functools import lru_cache
from pathlib import Path

import numpy as np

from game import Context, FULL, SIZE, Solver, TYPES, select_action

HERE = Path(__file__).resolve().parent
FIT_SEEDS = tuple(range(8141, 8146))
CONTROLS = ("qmdp", "depth3", "depth5", "depth8", "match_current", "persistent_high")
WIDTH = 64
STEPS = 6000
N_TEST = 400


def dump(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def contexts(role, seed, n):
    rng = np.random.default_rng(seed)
    horizons = (12, 16, 20) if role == "training" else (16, 20)
    return [Context(
        int(rng.choice(horizons)), float(rng.uniform(.94, .99)),
        float(rng.uniform(.12, .60)), float(rng.uniform(.04, .45)),
        float(rng.uniform(.015, .30)), float(rng.choice((.25, .5, .75))),
        tuple(rng.dirichlet(np.full(len(TYPES), rng.uniform(.35, 3.0)))),
    ) for _ in range(n)]


def arrays(model, x):
    w1, b1, w2, b2, w3, b3 = model
    first = np.tanh(x @ w1 + b1)
    second = np.tanh(first @ w2 + b2)
    return first, second, second @ w3 + b3


def train(x, y, seed):
    rng = np.random.default_rng(seed)
    parameters = [
        rng.normal(0, np.sqrt(1 / x.shape[1]), (x.shape[1], WIDTH)), np.zeros(WIDTH),
        rng.normal(0, np.sqrt(1 / WIDTH), (WIDTH, WIDTH)), np.zeros(WIDTH),
        rng.normal(0, np.sqrt(1 / WIDTH), (WIDTH, 1)), np.zeros(1),
    ]
    first_moment = [np.zeros_like(p) for p in parameters]
    second_moment = [np.zeros_like(p) for p in parameters]
    for step in range(1, STEPS + 1):
        index = rng.integers(len(x), size=256)
        batch, target = x[index], y[index, None]
        first, second, predicted = arrays(parameters, batch)
        d3 = (predicted - target) / len(batch)
        d2 = (d3 @ parameters[4].T) * (1 - second**2)
        d1 = (d2 @ parameters[2].T) * (1 - first**2)
        gradients = [batch.T @ d1, d1.sum(0), first.T @ d2, d2.sum(0),
                     second.T @ d3, d3.sum(0)]
        rate = .0015 * (.25 + .75 * (1 - step / STEPS))
        for p, grad, m, v in zip(parameters, gradients, first_moment, second_moment):
            grad += 1e-5 * p
            m *= .9
            m += .1 * grad
            v *= .999
            v += .001 * grad**2
            p -= rate * (m / (1 - .9**step)) / (np.sqrt(v / (1 - .999**step)) + 1e-8)
    return parameters


def feature_matrix(solver, states, representation):
    method = solver.features if representation == "full" else solver.prediction_features
    x = np.stack([method(state) for state in states])
    if representation == "mean":
        x = np.column_stack((x[:, :2], np.zeros((len(x), len(TYPES) - 2)), x[:, 2:]))
    return x


def table(solver, representation, model):
    states = list(solver.records)
    x = feature_matrix(solver, states, representation)
    predicted = arrays(model, x)[-1].ravel()
    return dict(zip(states, (predicted > 0).astype(int)))


def action(solver, name, state):
    remaining, b, y, g, mask = state
    if name == "match_current":
        return y
    if name == "persistent_high":
        return g
    if name.startswith("depth"):
        return select_action(solver.limited_values(min(remaining, int(name[5:])), b, y, g, mask))
    if name == "qmdp":
        probabilities = solver.belief(mask)
        values = sum(probabilities[i] * np.array(solver.values(remaining, b, y, g, 1 << i))
                     for i in np.flatnonzero(probabilities))
        return select_action(values)
    raise ValueError(name)


def policy_score(solver, chooser):
    @lru_cache(None)
    def value(remaining, b, y, g, mask):
        if remaining == 0:
            return 0.
        state = remaining, b, y, g, mask
        a = chooser(state)
        return solver.reward(a, y, g) + solver.context.discount * sum(
            p * value(remaining - 1, nb, ny, ng, nm)
            for p, nb, ny, ng, nm in solver.children(b, y, g, mask, a)
        )
    n = solver.context.horizon
    return .5 * (value(n, 0, 0, 0, FULL) + value(n, 0, 0, 1, FULL)) / solver.normalize(n)


def solve(context):
    solver = Solver(context)
    for g in (0, 1):
        solver.values(context.horizon, 0, 0, g, FULL)
    return solver


def fit():
    target = HERE / "models.npz"
    if target.exists() and not (HERE / "development_v1" / "models.npz").exists():
        raise FileExistsError(target)
    if (HERE / "freeze.json").exists():
        raise FileExistsError("Already frozen")
    rng = np.random.default_rng(73124)
    full, means, values = [], [], []
    for context in contexts("training", 16351, 96):
        solver = solve(context)
        states = list(solver.records)
        selected = rng.choice(len(states), size=min(600, len(states)), replace=False)
        states = [states[i] for i in selected]
        full.append(feature_matrix(solver, states, "full"))
        means.append(feature_matrix(solver, states, "mean"))
        values.extend(solver.records[state][1] - solver.records[state][0] for state in states)
        solver.values.cache_clear()
        solver.limited_values.cache_clear()
    x_full, x_mean, y = np.concatenate(full), np.concatenate(means), np.array(values)
    saved = {}
    diagnostics = {}
    for seed in FIT_SEEDS:
        for representation, x in (("full", x_full), ("mean", x_mean)):
            model = train(x, y, seed)
            for i, parameter in enumerate(model):
                saved[f"{representation}_{seed}_{i}"] = parameter
            prediction = arrays(model, x)[-1].ravel()
            diagnostics[f"{representation}_{seed}"] = {
                "rmse": float(np.sqrt(np.mean((prediction - y)**2))),
                "action_match": float(np.mean((prediction > 0) == (y > 0))),
            }
        print("fit", seed, flush=True)
    np.savez_compressed(target, **saved)
    dump(HERE / "fit_diagnostics.json", {"examples": len(y), "fits": diagnostics})


def load_models():
    with np.load(HERE / "models.npz") as archive:
        return {(representation, seed): [archive[f"{representation}_{seed}_{i}"] for i in range(6)]
                for representation in ("full", "mean") for seed in FIT_SEEDS}


def evaluate(bank, controls):
    models = load_models()
    names = ("full", "mean", *controls, "exact")
    scores = np.empty((len(names), len(FIT_SEEDS), len(bank)))
    gaps = np.empty((2, len(FIT_SEEDS), len(bank)))
    for j, context in enumerate(bank):
        solver = solve(context)
        root_score = .5 * sum(max(solver.values(context.horizon, 0, 0, g, FULL)) for g in (0, 1))
        scores[-1, :, j] = root_score / solver.normalize(context.horizon)
        for i, seed in enumerate(FIT_SEEDS):
            for p, rep in enumerate(("full", "mean")):
                decisions = table(solver, rep, models[rep, seed])
                scores[p, i, j] = policy_score(solver, decisions.__getitem__)
                gaps[p, i, j] = max(
                    max(q) - q[decisions[state]] for state, q in list(solver.records.items())
                    if state in decisions
                )
        for p, name in enumerate(controls, 2):
            scores[p, :, j] = policy_score(solver, lambda state: action(solver, name, state))
        solver.values.cache_clear()
        solver.limited_values.cache_clear()
        if (j + 1) % 40 == 0:
            print("evaluated", j + 1, flush=True)
    return names, scores, gaps


def development():
    bank = contexts("development", 86417, 40)
    names, scores, _ = evaluate(bank, CONTROLS)
    np.savez_compressed(HERE / "development_scores.npz", names=names, scores=scores)
    summary = dict(zip(names, scores.mean(axis=(1, 2)).tolist()))
    dump(HERE / "development.json", summary)


def freeze():
    if (HERE / "freeze.json").exists():
        raise FileExistsError("Already frozen")
    development = json.loads((HERE / "development.json").read_text())
    best = max(CONTROLS, key=development.__getitem__)
    files = ("game.py", "run.py", "PROTOCOL.md", "models.npz", "development.json",
             "development_scores.npz", "fit_diagnostics.json", "audit_oracle.py", "exact_witness.py")
    dump(HERE / "freeze.json", {"selected_control": best,
                                "hashes": {name: sha(HERE / name) for name in files}})


def analysis(names, scores):
    rng = np.random.default_rng(41399)
    n, fits, bases = scores.shape
    delta = scores[:1] - scores[1:-1]
    draws = np.empty((10000, len(delta)))
    for i in range(len(draws)):
        fit_index = rng.integers(fits, size=fits)
        base_index = rng.integers(bases, size=bases)
        draws[i] = delta[:, fit_index][:, :, base_index].mean(axis=(1, 2))
    limits = np.quantile(draws, [.0125, .9875], axis=0)
    exact = scores[-1]
    gaps = exact - scores[0]
    return {
        "payoff": dict(zip(names, scores.mean(axis=(1, 2)).tolist())),
        "full_policy": {"mean_gap_to_exact": float(gaps.mean()), "worst_environment_fit_gap": float(gaps.max()),
                        "fraction_within_one_point": float(np.mean(gaps <= .01)),
                        "by_fit_gap": gaps.mean(axis=1).tolist()},
        "comparisons": {name: {"difference": float(d.mean()), "simultaneous_95_interval": ci.tolist()}
                        for name, d, ci in zip(names[1:-1], delta, limits.T)},
        "environments": bases, "fits": fits,
    }


def confirmation(replay=False):
    receipt = json.loads((HERE / "freeze.json").read_text())
    for name, expected in receipt["hashes"].items():
        if sha(HERE / name) != expected:
            raise ValueError(f"Changed frozen file {name}")
    path = HERE / "confirmation_bank.json"
    if replay:
        bank = [Context(**{**c, "prior": tuple(c["prior"])}) for c in json.loads(path.read_text())]
        provenance = json.loads((HERE / "bank_receipt.json").read_text())
        assert sha(path) == provenance["bank_sha256"]
        assert sha(HERE / "freeze.json") == provenance["freeze_sha256"]
        assert bank == contexts("confirmation", provenance["seed"], N_TEST)
    else:
        if path.exists():
            raise FileExistsError(path)
        seed = secrets.randbits(63)
        bank = contexts("confirmation", seed, N_TEST)
        dump(path, [context.__dict__ for context in bank])
        dump(HERE / "bank_receipt.json", {"seed": seed, "bank_sha256": sha(path),
                                         "freeze_sha256": sha(HERE / "freeze.json")})
    names, scores, gaps = evaluate(bank, (receipt["selected_control"],))
    result = analysis(names, scores)
    assert np.min(scores[-1] - scores[:-1]) > -1e-10
    if replay:
        with np.load(HERE / "confirmation_scores.npz") as source:
            np.testing.assert_array_equal(scores, source["scores"])
            np.testing.assert_array_equal(gaps, source["max_bellman_gaps"])
        assert result == json.loads((HERE / "confirmation.json").read_text())
        print("Exact replay passed")
    else:
        np.savez_compressed(HERE / "confirmation_scores.npz", names=names, scores=scores, max_bellman_gaps=gaps)
        dump(HERE / "confirmation.json", result)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=("fit", "development", "freeze", "confirmation", "replay"))
    phase = p.parse_args().phase
    if phase == "replay":
        confirmation(True)
    else:
        globals()[phase]()
