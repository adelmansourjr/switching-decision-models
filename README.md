# Switching Dynamics and Coordination Games

I studied how information about hidden switching dynamics changes decisions in a small coordination game. The project contains exact counterexamples, a planning oracle, simulated measurements and a learned policy evaluated on frozen generated contexts.

[Relay results](RESULTS.md) · [Game report](equilibrium_pilot/REPORT.md) · [Exact game witness](equilibrium_pilot/MATHEMATICS.md)

The relay experiment found a phase effect in two withheld timing families. The active policy missed its payoff criterion. In the game study, compressing the opponent distribution to average thresholds can change the optimal first action. The report also gives the learned policy's remaining regret against exact planning.

Use Python 3.12 and NumPy. Matplotlib is only needed to regenerate figures.

```sh
python check_relay_seed.py
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python relay_study.py replay
cd equilibrium_pilot
python exact_witness.py
python audit_oracle.py
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python run.py replay
```

The code, fixed simulation banks, numeric policy parameters and archived results are included. The experiments measure simulated behaviour under the specified switching and opponent models.
