"""Self-contained pilot on phase interventions and the net value of measuring phase."""

import argparse
import hashlib
import json
import secrets
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FAMILIES = {"fit": ("linear", "sinusoidal"), "development": ("linear", "sinusoidal"),
            "confirmation": ("chirp", "rate_break")}
GRID = np.arange(96) / 96
COST = .008
SIGMA_COARSE, SIGMA_FINE = .14, .026
N_CONFIRM = 1000
SEEDS = {"fit": 21753, "development": 24087}
DRAWS = 10000
CAP = .35
N_PHASE, N_TIME = len(GRID), 28


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bank(role, family, size, seed):
    family_id = ("linear", "sinusoidal", "chirp", "rate_break").index(family)
    if family not in FAMILIES[role]:
        raise ValueError("Mechanism is unavailable for this role")
    rng = np.random.default_rng(np.random.SeedSequence([seed, family_id]))
    return dict(
        cycles=rng.integers(24, 91, size), duty=rng.uniform(.34, .66, (size, 2)),
        drift=rng.uniform(-.45, .45, size),
        amplitude=rng.uniform(-.16, .16, size),
        phase=rng.random(size), shift=rng.uniform(.045, .34, size),
        shift_cost=rng.uniform(.025, .18, size),
        coarse=rng.normal(0, SIGMA_COARSE, size), fine=rng.normal(0, SIGMA_FINE, size),
        lottery=rng.random(size),
    )


def shape(time, family):
    if family == "linear":
        return np.zeros_like(time)
    if family == "sinusoidal":
        return np.sin(2 * np.pi * time)
    if family == "chirp":
        return 4 * time * (1 - time)
    if family == "rate_break":
        return 1 - np.abs(2 * time - 1)
    raise ValueError(family)


def true_occupancy(data, family, phase, steps=36, near_sync=False):
    """Find all switch times by monotone inversion, then integrate exact step intervals."""
    joint, one, two = (np.empty(len(phase)) for _ in range(3))
    for i, (cycles, (duty1, duty2), drift, amp, offset) in enumerate(
        zip(data["cycles"], data["duty"], data["drift"], data["amplitude"], phase % 1)
    ):
        if near_sync:
            drift *= .25
        candidates = np.arange(-1, cycles + 3)[:, None] + np.array([0, duty2])
        levels = candidates[(candidates > offset) & (candidates < cycles + drift + offset)]
        lower, upper = np.zeros(len(levels)), np.ones(len(levels))
        for _ in range(steps):
            middle = (lower + upper) / 2
            below = (cycles + drift) * middle + amp * shape(middle, family) + offset < levels
            lower = np.where(below, middle, lower)
            upper = np.where(below, upper, middle)
        second_cuts = (lower + upper) / 2
        first_cuts = (np.arange(cycles)[:, None] + np.array([0, duty1])).ravel() / cycles
        cuts = np.sort(np.r_[first_cuts, second_cuts, 1.])
        centers = (cuts[:-1] + cuts[1:]) / 2
        durations = np.diff(cuts)
        first = cycles * centers % 1 < duty1
        second = ((cycles + drift) * centers + offset + amp * shape(centers, family)) % 1 < duty2
        joint[i], one[i], two[i] = durations @ (first & second), durations @ first, durations @ second
    return np.column_stack((joint, one, two))


def circular_overlap(duty1, duty2, phase):
    phase = phase % 1
    return sum(np.maximum(0, np.minimum(duty1, k + duty2 - phase) - np.maximum(0, k - phase))
               for k in (0., 1.))


def physics(data, phase_grid, shift=0, near_sync=False):
    """Coarse predictor from the duty fractions and average phase drift."""
    duty1, duty2 = data["duty"].T
    drift = data["drift"] * (.25 if near_sync else 1)
    phase = phase_grid + shift
    time = (np.arange(N_TIME) + .5) / N_TIME
    phase = phase[..., None] + drift[:, None, None] * time
    return circular_overlap(duty1[:, None, None], duty2[:, None, None], phase).mean(-1)


def design(data):
    d1, d2 = data["duty"].T - .5
    drift = data["drift"]
    return np.column_stack((np.ones(len(drift)), d1, d2, drift, drift**2,
                            d1 * drift, d2 * drift, d1 * d2, d1**2, d2**2))


def harmonics(phase):
    return np.concatenate((np.ones(phase.shape + (1,)),
                           np.cos(2 * np.pi * phase[..., None] * np.arange(1, 6)),
                           np.sin(2 * np.pi * phase[..., None] * np.arange(1, 6))), axis=-1)


def learn():
    inputs, targets = [], []
    for family in FAMILIES["fit"]:
        data = bank("fit", family, 1050, SEEDS["fit"])
        context = design(data)
        for phase in (data["phase"], data["phase"] + data["shift"]):
            y = true_occupancy(data, family, phase)[:, 0]
            x = context[:, :, None] * harmonics(phase)[:, None, :]
            inputs.append(x.reshape(len(y), -1))
            targets.append(y)
    x, y = np.concatenate(inputs), np.concatenate(targets)
    regularizer = np.eye(x.shape[1]) * .08
    regularizer[0, 0] = 0
    return np.linalg.solve(x.T @ x + regularizer, x.T @ y).reshape(10, 11)


def learned(data, phase_grid, coefficients):
    weight = design(data) @ coefficients
    return np.einsum("bf,bpf->bp", weight, harmonics(phase_grid))


def wrapped_normal(delta, sigma):
    return sum(np.exp(-.5 * ((delta + k) / sigma)**2) for k in (-1, 0, 1))


def measurements(data):
    before = (data["phase"] + data["coarse"]) % 1
    after = (data["phase"] + data["fine"]) % 1
    prior = wrapped_normal(before[:, None] - GRID, SIGMA_COARSE)
    prior /= prior.sum(1, keepdims=True)
    posterior = prior * wrapped_normal(after[:, None] - GRID, SIGMA_FINE)
    posterior /= posterior.sum(1, keepdims=True)
    return prior, posterior


def valuation(gain, prior, posterior):
    likelihood = wrapped_normal(GRID[:, None] - GRID, SIGMA_FINE)
    likelihood /= likelihood.sum(1, keepdims=True)
    blind = (prior * gain).sum(1)
    informed = (posterior * gain).sum(1)
    weighted = (prior * gain) @ likelihood
    info_value = np.maximum(weighted, 0).sum(1) - np.maximum(blind, 0)
    return blind, informed, info_value


def select(priority, count):
    result = np.zeros(len(priority), dtype=bool)
    result[np.argsort(-priority, kind="stable")[:count]] = True
    return result


def evaluate(data, family, coefficients):
    shift, shift_cost = data["shift"][:, None], data["shift_cost"][:, None]
    # Separate the two arm predictions before taking the shared-phase contrast.
    gain = physics(data, GRID, shift) - physics(data, GRID) - shift_cost
    learned_gain = learned(data, GRID[None, :] + shift, coefficients) - learned(
        data, np.broadcast_to(GRID, gain.shape), coefficients) - shift_cost
    prior, posterior = measurements(data)
    blind, informed, value = valuation(gain, prior, posterior)
    rival_blind, rival_informed, rival_value = valuation(learned_gain, prior, posterior)
    quantity = min(int(CAP * len(value)), np.count_nonzero(value > COST))
    physics_probe = select(value, quantity)
    actual = true_occupancy(data, family, data["phase"])[:, 0]
    changed = true_occupancy(data, family, data["phase"] + data["shift"])[:, 0]
    actual_gain = changed - actual - data["shift_cost"]
    baseline = {
        "targeted_physics": (physics_probe, blind, informed),
        "margin": (select(-np.abs(blind), quantity), blind, informed),
        "learned_value": (select(rival_value, quantity), rival_blind, rival_informed),
        "random_equal_budget": (select(data["lottery"], quantity), blind, informed),
        "no_measurement": (np.zeros(len(value), dtype=bool), blind, informed),
        "measure_everywhere": (np.ones(len(value), dtype=bool), blind, informed),
    }
    outcomes = {name: actual_gain * (np.where(probe, full, cheap) > 0) - COST * probe
                for name, (probe, cheap, full) in baseline.items()}
    low = true_occupancy(data, family, np.full(len(actual), .5), near_sync=True)
    high = true_occupancy(data, family, np.zeros(len(actual)), near_sync=True)
    d1, d2 = data["duty"].T
    before_prediction = physics(data, np.zeros((len(actual), 1)), near_sync=True)[:, 0]
    after_prediction = physics(data, np.full((len(actual), 1), .5), near_sync=True)[:, 0]
    safe = d1 * d2
    return {
        "outcomes": outcomes, "probe_fraction": float(physics_probe.mean()),
        "headroom": np.maximum(actual_gain, 0),
        "physics_probe": physics_probe,
        "intervention": np.column_stack((high - low, high[:, 0] > safe, low[:, 0] < safe)),
        "physics_error": (np.abs(high[:, 0] - before_prediction) + np.abs(low[:, 0] - after_prediction)) / 2,
        "independent_error": (np.abs(high[:, 0] - safe) + np.abs(low[:, 0] - safe)) / 2,
        "actual_gain": actual_gain,
    }


def aggregate(results, controls):
    families = tuple(results)
    rng = np.random.default_rng(307109)
    keys = ("joint_high_change", "relay1_duty_change", "relay2_duty_change", "reversal_fraction",
            "physics_mae", "independent_mae", *(f"net_vs_{control}" for control in controls))
    series = []
    for r in results.values():
        inter = r["intervention"]
        comparison = [r["outcomes"]["targeted_physics"] - r["outcomes"][control] for control in controls]
        series.append(np.column_stack((inter[:, :3], inter[:, 3] * inter[:, 4], r["physics_error"],
                                       r["independent_error"], *comparison)))
    draws = np.zeros((DRAWS, len(keys)))
    for a in series:
        for start in range(0, DRAWS, 250):
            n = min(DRAWS - start, 250)
            idx = rng.integers(len(a), size=(n, len(a)))
            draws[start:start + n] += a[idx].mean(1) / len(families)
    mean = np.stack([a.mean(0) for a in series]).mean(0)
    limits = np.quantile(draws, [.0125, .9875], axis=0).T
    endpoints = {key: {"mean": float(m), "interval": ci.tolist()}
                 for key, m, ci in zip(keys, mean, limits)}
    policies = {p: float(np.mean([r["outcomes"][p].mean() for r in results.values()]))
                for p in next(iter(results.values()))["outcomes"]}
    return {"endpoints": endpoints, "net_gain_over_never_shifting": policies,
            "families": {f: {"probe_fraction": r["probe_fraction"],
                             "headroom": float(r["headroom"].mean()),
                             "reversal_fraction": float((r["intervention"][:, 3] *
                                                         r["intervention"][:, 4]).mean())}
                         for f, r in results.items()}}


def freeze():
    if (HERE / "freeze.json").exists():
        raise FileExistsError("Existing experiment freeze")
    coefficients = learn()
    np.save(HERE / "coefficients.npy", coefficients)
    dev = {family: evaluate(bank("development", family, 320, SEEDS["development"]), family, coefficients)
           for family in FAMILIES["development"]}
    write_json(HERE / "development.json", aggregate(dev, ("margin", "learned_value")))
    frozen = ("relay_study.py", "PROTOCOL.md", "coefficients.npy", "development.json")
    write_json(HERE / "freeze.json", {name: sha(HERE / name) for name in frozen})


def verify_freeze():
    receipt = json.loads((HERE / "freeze.json").read_text())
    for filename, expected in receipt.items():
        if sha(HERE / filename) != expected:
            raise ValueError(f"Frozen file changed: {filename}")


def confirm(replay=False):
    verify_freeze()
    score_path = HERE / "confirmation_scores.npz"
    receipt_path = HERE / "bank_receipt.json"
    bank_path = HERE / "confirmation_bank.npz"
    if replay:
        receipt = json.loads(receipt_path.read_text())
        if sha(bank_path) != receipt["bank_sha256"] or sha(HERE / "freeze.json") != receipt["freeze_sha256"]:
            raise ValueError("Bank provenance verification failed")
        seed = receipt["seed"]
    else:
        if any(path.exists() for path in (score_path, receipt_path, bank_path)):
            raise FileExistsError("Existing confirmation bank or scores")
        seed = secrets.randbits(63)
    banks = {f: bank("confirmation", f, N_CONFIRM, seed) for f in FAMILIES["confirmation"]}
    saved_bank = {f"{f}_{key}": x for f, data in banks.items() for key, x in data.items()}
    if replay:
        with np.load(bank_path) as archived:
            for key, value in saved_bank.items():
                np.testing.assert_array_equal(value, archived[key])
    else:
        np.savez_compressed(bank_path, **saved_bank)
        write_json(receipt_path, {"seed": seed, "bank_sha256": sha(bank_path),
                                  "freeze_sha256": sha(HERE / "freeze.json")})
    coeff = np.load(HERE / "coefficients.npy")
    results = {f: evaluate(data, f, coeff) for f, data in banks.items()}
    summary = aggregate(results, ("margin", "learned_value"))
    arrays = {f"{f}_{key}": value for f, result in results.items()
              for key, value in result.items() if key != "outcomes"}
    arrays.update({f"{f}_net_{key}": value for f, r in results.items() for key, value in r["outcomes"].items()})
    if replay:
        with np.load(score_path) as archived:
            for key, value in arrays.items():
                np.testing.assert_array_equal(value, archived[key])
        if summary != json.loads((HERE / "confirmation.json").read_text()):
            raise ValueError("Analysis changed on replay")
        print("Bank, scenario scores and analysis replayed exactly.")
    else:
        np.savez_compressed(score_path, **arrays)
        write_json(HERE / "confirmation.json", summary)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("freeze", "confirm", "replay"))
    phase = parser.parse_args().phase
    if phase == "freeze":
        freeze()
    else:
        confirm(phase == "replay")
