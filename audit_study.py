"""Numerical audits separate from the frozen confirmation analysis."""

import json
from fractions import Fraction

import numpy as np

import relay_study as study
from check_relay_seed import integrated_overlap, overlap


def main():
    rng = np.random.default_rng(3310)
    errors = []
    with np.load(study.HERE / "confirmation_bank.npz") as bank:
        for family in study.FAMILIES["confirmation"]:
            rows = rng.choice(study.N_CONFIRM, 96, replace=False)
            data = {key[len(family) + 1:]: bank[key][rows] for key in bank.files
                    if key.startswith(family + "_")}
            for phase in (data["phase"], data["phase"] + data["shift"]):
                ordinary = study.true_occupancy(data, family, phase)
                refined = study.true_occupancy(data, family, phase, steps=52)
                errors.append(float(np.abs(ordinary - refined).max()))
    assert max(errors) < 1e-7
    grid = np.arange(2000) / 2000
    checks = 0
    for i in (3, 6, 9, 10, 13, 17):
        for j in (3, 6, 9, 10, 13, 17):
            a, b = i / 20, j / 20
            values = study.circular_overlap(a, b, grid)
            np.testing.assert_allclose(values.mean(), a * b, atol=1e-14)
            for shift in (.05, .1, .25, .35, .5):
                difference = study.circular_overlap(a, b, grid + shift) - values
                bound = min(shift, a, b, 1 - a, 1 - b)
                np.testing.assert_allclose([difference.min(), difference.max()], [-bound, bound], atol=1e-14)
                checks += 1
    for phase in [Fraction(0), Fraction(1, 7), Fraction(1, 2)]:
        exact = np.array([float(overlap(phase + Fraction(i, 20))) for i in range(20)])
        np.testing.assert_allclose(study.circular_overlap(.5, .5, float(phase) + np.arange(20) / 20), exact)
    assert integrated_overlap(Fraction(0), Fraction(1, 2)) == Fraction(1, 4)
    result = {"refined_event_scenarios": 384, "max_refinement_error": max(errors),
              "unequal_duty_grid_checks": checks}
    (study.HERE / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()
