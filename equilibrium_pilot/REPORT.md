# Exact decision limit and compact learned policy in a switching coordination game

Adel Mansour · 12 November 2026

The new experiment advances the proposal in two ways. An exact counterexample proves that mean activation and release thresholds can lose information needed for the optimal first action. A 5,441-parameter learned policy gets close to the exact planning ceiling on fresh parameter draws and has substantially lower measured single-query CPU latency than a depth-eight planner. The learned policy earned slightly less payoff than that planner and its advantage over a mean-threshold network was uncertain.

## What was tested

The finite game has two players, five possible private hysteresis types, and a known public belief-update rule on a nine-point grid. Each stage game has two strict coordinated Nash equilibria. Public context changes which one pays more, so the focal policy learns to track a desirable equilibrium and account for transition costs. It optimizes its payoff against the opponent's declared myopic switching rule. The opponent does not solve an unrestricted forward-looking dynamic game.

The focal policy knows the prior over the five types and observes completed context, memory, and opponent action. Exact filtering is supplied to every planning policy. The network receives this posterior and four action-conditioned probabilities calculated from the known threshold dynamics. Its value network is trained on 55,288 exact action-value labels from 96 training environments. The mean-threshold network gets the same public context, filter, label set, capacity and training budget, but its value input retains only mean thresholds and plug-in transition predictions. The optimal policy shares the same observations and prior. It has unlimited planning computation.

A first development pass used a smaller network without explicit switch features and scored 0.76879 against 0.77810 for the exact teacher. A second pass widened both networks, added switch features and doubled both training budgets. These choices were made on development data. The improved fit cannot be attributed solely to the new features because size and training experience changed together. Both primary networks in the final comparison have the same size and experience. The first pass is retained in development_v1.

A local source and fit freeze preceded generation of the 400-context confirmation bank. Payoffs, discount, prior weights and switching frequency were freshly drawn. The same five opponent types and public transition family were available in training. The test measures interpolation over new parameter draws within that declared family. Five independent fit seeds were paired across the learned models. Return is evaluated exactly over private type and future public contexts. [Frozen protocol](PROTOCOL.md) · [Freeze receipt](freeze.json) · [Bank receipt](bank_receipt.json)

## Exact counterexample

Two opponent priors have the same mean release threshold, 1.8, and activation threshold, 4.2. Their optimal first actions are opposite. The game has horizon eight, discount .97, lower coordinated reward .46, mismatch loss .45, public context flip probability .15 and memory rate .5. The first-action gaps, including exact optimal continuation, are +1.0997 and −1.8100 payoff points.

If a first-action policy retains only those two means, it must use the same randomized action in both cases. Its best worst-case loss relative to the exact policy is at least **0.68409 payoff points**. Rational arithmetic verifies both signs and the lower bound, without floating-point tolerance. The restriction applies to the compressed first-decision input. It does not apply to a policy given the full prior or extra informative observations.

This witness was found by exploratory search, then rounded to simple rational parameters and independently verified. It provides an exact mathematical example rather than an estimate of how often the issue occurs. Belief sufficiency is established theory. Priority for any more general threshold-specific theorem remains a research question. [Mathematics and proof](MATHEMATICS.md) · [Rational checker](exact_witness.py) · [Exact fractions](exact_witness.json)

## Held-out payoff and optimality gap

One payoff point is .01 of the time-normalized discounted return. Maximum stage reward is 1, while mismatches can earn a negative payoff. Scores below are payoff values, not classification accuracy.

| Policy | Mean normalized payoff | Gap to exact optimum, points |
|---|---:|---:|
| Exact public-information planner | 0.775367 | 0 |
| Depth-eight planner | 0.773839 | 0.153 |
| Full-posterior learned policy | 0.772542 | 0.282 |
| Mean-threshold learned policy | 0.771736 | 0.363 |

The full-posterior network's gap to the exact optimum averaged 0.282 points across 400 independent contexts and five fits. It stayed within one point in 91.8% of the 2,000 context-by-fit evaluations. The worst gap was **6.839 points**. Thus the small average gap coexists with consequential failures in some configurations.

| Primary paired contrast | Mean difference, points | Simultaneous 95% interval, points |
|---|---:|---:|
| Full posterior minus mean thresholds | +0.081 | [−0.045, +0.246] |
| Full posterior minus depth eight | −0.130 | [−0.218, −0.038] |

Intervals resample paired fit seeds and independent contexts as crossed factors. The practical gate required both lower bounds to exceed +1 point. It failed on both contrasts. The exact counterexample establishes that mean compression can fail, while the held-out comparison leaves the full network's average gain over the mean network uncertain. [Machine-readable analysis](confirmation.json) · [Scenario scores](confirmation_scores.npz)

## Decision time

On 144 development histories with at least eight rounds left, repeated five times, the median mean single-query CPU time was **19.1 microseconds** for the full learned policy and **1.141 milliseconds** for depth-eight planning, a ratio of **59.7**. Each planner query used a fresh lookahead cache. Timings include the learned policy's feature computation. They exclude common filter/model setup and all training and teacher costs. They are a local hardware diagnostic, outside the frozen primary efficacy tests. Batching, persistent planner caching and other implementations can change this ratio. The network's slightly lower payoff accompanies this observed reduction in inference time. [Timing script](latency.py) · [Timing results](latency.json)

## Verification and interpretation

The independent short-horizon enumerator agreed with the cached oracle on 48 cases to within \(1.78\times10^{-15}\), and five follower payoff/switching-cost derivations reproduce the supported threshold pairs. The counterexample uses exact fractions. The confirmation archive includes expected returns for every context and fit, with reproducible seeds and source hashes.

The result is a stronger computational starting point for the proposal. It supplies a mathematical limit, a trained policy and an exact way to measure its remaining loss. It leaves superiority over the strongest planner, architectural necessity, unseen mechanism transfer and physical validation open. The challenge for the next study is to reduce the costly worst cases and test whether the speed–payoff trade-off persists under fair alternative implementations.
