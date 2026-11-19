# Simulation results and evidence limits

Adel Mansour · 11 November 2026

The phase-reset experiment passed all three prespecified mechanistic criteria. The phase-sensing policy fell short of its separate practical payoff criterion. This is a computational feasibility study with two withheld timing mechanisms.

## Design

The frozen experiment used 2,000 independently drawn synthetic scenarios, 1,000 per timing family. The new families were a smooth rate sweep and an abrupt mid-window rate change. Both preserve the calibrated mean phase drift, which the predictor uses. They differ in the unobserved path by which the second relay accumulates phase. Both channels were assigned independently drawn duty cycles from 34% to 66%. The policies saw a coarse phase reading, the calibrated duty cycles and mean drift, and the shift size and price. The fine phase reading incurred a charge. No policy saw the timing family or the within-window timing perturbation. A separate paired phase-reset experiment manipulated the initial relative phase and used a narrower drift range. [Frozen protocol](PROTOCOL.md) · [Bank receipt](bank_receipt.json)

## Mechanistic endpoint

| Paired effect of moving from half-cycle separation to synchrony | Estimate | 97.5% marginal interval |
|---|---:|---:|
| Time both channels were high | +31.782 pp | +31.367 to +32.194 |
| Relay 1 duty fraction | 0.000 pp | 0.000 to 0.000 |
| Relay 2 duty fraction | +0.071 pp | +0.067 to +0.076 |
| Scenarios with the predicted two-sided action reversal | 99.90% | 99.70% to 100.00% |

Both single-channel intervals were inside the predeclared ±0.50-point equivalence margin. Channel 1 used an integer number of cycles, so its measured duty fraction was fixed up to arithmetic error. The small shift in measured channel 2 duty fraction arises because its rate changes and its time window may truncate a cycle. Thus the evidence supports practical equivalence under this tolerance rather than exact equality in every finite trial. The reference decision compares joint-high payoff with a scenario-specific fixed payoff equal to the product of the calibrated duty fractions. Reversals occurred in 998 of 1,000 rate-sweep cases and 1,000 of 1,000 rate-break cases. The two exceptions are retained.

A model using duty fractions, phase and average drift predicted joint occupancy in the two phase conditions with mean absolute error **2.54 points**. The independent product model's error was **15.89 points**. The phase condition was assigned and known for this prediction. The comparison measures the value of timing information in this test, and is distinct from predicting an unobserved natural phase.

## Active sensing endpoint

All equal-budget policies bought 350 measurements per family and paid the same fee, 0.008 per purchase. Values below are net of that charge. A learned Fourier comparator trained on 2,100 synthetic configurations with both arm labels and true phase available during training. Both the learned and physics policies used the same calibrated sensor likelihood in evaluation.

| Policy | Net payoff gain over never shifting, pp | Physics minus control, pp | 95% simultaneous intervals for the two primary contrasts |
|---|---:|---:|---:|
| Targeted physics model | +1.285 | | |
| Smallest blind margin first | +1.170 | +0.115 | +0.037 to +0.200 |
| Learned Fourier value policy | +1.308 | −0.023 | −0.101 to +0.051 |
| Random allocation, equal budget | +0.810 | | |
| No fine measurement | +0.860 | | |
| Measure everywhere | +0.927 | | |

The target required both primary lower bounds to exceed +0.20 points. Neither did. The learned policy and physics policy were close under the declared budget. The physics policy's descriptive gain over no fine measurement was +0.425 points. Full foreknowledge of each shift's realized payoff before choosing, with no sensing charge, gives a gross diagnostic headroom of +2.089 points over never shifting. That quantity includes unavailable information and does not establish a public-information optimum.

## Verification and scope

All intervals use a paired, stratified bootstrap over configurations. They quantify variation over draws from these two fixed synthetic generators. The local freeze retains the fitted baseline, development summary and a source hash. The bank is archived with its seed and was replayed exactly on the same numerical platform. An independent seed checker verified 80 finite-window cases and 132 exact value-of-information cases. An unequal-duty audit checked 180 fixed-phase configurations. Increasing the event solver from 36 to 52 bisections in 384 arm evaluations changed a reported occupancy by at most \(4.58\times10^{-10}\).

This experiment confirms a controlled timing effect in the simulator. Its phase intervention is intentionally strong, the sensing model is calibrated by construction, and the scoring function rewards a prescribed joint event. A physical relay test would establish whether the effect survives clock, actuator and measurement imperfections. A repeated game is a further test, because other players may respond to interventions and decision makers can learn from past outcomes. These remain thesis work. [Mathematics](EXTENSION.md) · [Numerical audit](audit.json) · [Full machine-readable results](confirmation.json)
