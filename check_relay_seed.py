"""Independently enumerate relay events and check the finite-window formula."""

from fractions import Fraction
from math import ceil, floor, isclose


def high_fraction_by_events(cycles, other_cycles, phase):
    cuts = {Fraction(0), Fraction(1)}
    for rate, start in ((cycles, 0), (other_cycles, phase)):
        for j in range(ceil(2 * start), floor(2 * (start + rate)) + 1):
            cuts.add((Fraction(j, 2) - start) / rate)
    cuts = sorted(cuts)
    high = Fraction(0)
    for left, right in zip(cuts, cuts[1:]):
        midpoint = (left + right) / 2
        if cycles * midpoint % 1 < Fraction(1, 2) and (other_cycles * midpoint + phase) % 1 < Fraction(1, 2):
            high += right - left
    return high


def overlap(phase):
    location = phase % 1
    return Fraction(1, 2) - min(location, 1 - location)


def integrated_overlap(phase, drift):
    if not drift:
        return overlap(phase)
    lo, hi = sorted((phase, phase + drift))
    knots = [lo] + [Fraction(j, 2) for j in range(floor(2 * lo) + 1, ceil(2 * hi))] + [hi]
    integral = sum((right - left) * (overlap(left) + overlap(right)) / 2
                   for left, right in zip(knots, knots[1:]))
    return integral / (hi - lo)


def phase_information_by_segments(shift, cost):
    knots = sorted({Fraction(0), Fraction(1, 2), Fraction(1), (1 - shift) % 1, (Fraction(1, 2) - shift) % 1})
    gain = lambda phase: overlap(phase + shift) - overlap(phase) - cost
    value = Fraction(0)
    for left, right in zip(knots, knots[1:]):
        a, b = gain(left), gain(right)
        if a >= 0 and b >= 0:
            value += (right - left) * (a + b) / 2
        elif a * b < 0:
            value += (right - left) * max(a, b) ** 2 / (2 * abs(a - b))
    return value


def main():
    phase_values = [Fraction(0), Fraction(1, 7), Fraction(1, 2), Fraction(9, 10)]
    drift_values = [Fraction(0), Fraction(1, 8), Fraction(1, 2), Fraction(-3, 8), Fraction(9, 8)]
    cycle_values = [Fraction(4), Fraction(23, 2), Fraction(50), Fraction(100)]
    max_error = {}
    for cycles in cycle_values:
        errors = []
        for phase in phase_values:
            for drift in drift_values:
                actual = high_fraction_by_events(cycles, cycles + drift, phase)
                predicted = integrated_overlap(phase, drift)
                error = abs(actual - predicted)
                assert error <= (5 * abs(drift) + 1) / cycles
                errors.append(error)
        max_error[cycles] = max(errors)
    assert high_fraction_by_events(Fraction(50), Fraction(50), Fraction(0)) == Fraction(1, 2)
    assert high_fraction_by_events(Fraction(50), Fraction(50), Fraction(1, 2)) == 0
    assert integrated_overlap(Fraction(0), Fraction(1, 2)) == Fraction(1, 4)
    regret = min(max(.25 * (1 - i / 100), .25 * i / 100) for i in range(101))
    assert isclose(regret, .125)
    for i in range(11):
        shift = Fraction(i, 20)
        for j in range(12):
            cost = Fraction(j, 20)
            expected = max(0, shift - cost) * (1 - shift - cost) / 2
            assert phase_information_by_segments(shift, cost) == expected
    assert phase_information_by_segments(Fraction(1, 4), Fraction(1, 10)) == Fraction(39, 800)
    for cycles, error in max_error.items():
        print(f"{float(cycles):>5.1f} fast cycles: max absolute occupancy error {float(error):.6f}")
    print("80 finite-window checks passed. Sharp worst-case regret at the two endpoints: 0.125.")
    print("132 exact information-value checks passed. Example break-even observation cost: 0.04875.")


if __name__ == "__main__":
    main()
