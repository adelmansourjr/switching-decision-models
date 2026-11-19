# A decision-relevant limit on averaging hidden thresholds

Adel Mansour · 12 November 2026

## The declared game

The two players choose binary actions each round. The focal player earns 1 when both actions equal the public context, S when they coordinate on the other action, and -c when they mismatch. The follower has positive payoffs for both coordinated pairs and zero for mismatches. Hence both aligned pairs are strict Nash equilibria of the stage game, and one has higher payoff for both players. The private follower type controls activation and release thresholds in a known hysteresis rule. A switching charge derives those thresholds from a myopic payoff comparison. The public context changes over time, so the full decision problem tracks desirable stage-game equilibria and pays for each transition.

Let the hidden type be \(\theta=(l,u)\), with support

\[
(1,3),\ (1,5),\ (2,4),\ (2,5),\ (3,4).
\]

The public memory \(b\in\{0,\dots,8\}\) updates to \(\lfloor(1-\lambda)b+8\lambda a+.5\rfloor\). In public context 1, a follower in mode 0 changes to 1 at \(b\ge u\). A follower in mode 1 changes to 0 at \(b\le l\). In public context 0, replace the pair by \((8-u,8-l)\). The agent observes the completed memory, context and follower mode. It knows the prior, the observation rule and the transition class. The threshold realization is private.

For context 1, choose follower coordinated payoffs \(r=1-(l+u)/16\), \(s=(l+u)/16\) and switching cost \(\kappa=(u-l)/16\). A mode-0 follower switches when \(r\bar b\ge s(1-\bar b)+\kappa\), where \(\bar b=b/8\). This yields upper threshold \(u/8\). The reverse comparison yields \(l/8\). All supported pairs have \(r>s>0\). Reflecting the pair swaps \(r,s\), giving a higher payoff to both players for the other coordinated profile in public context 0.

## Exact optimal policy with public information

The sufficient state is \(h=(n,b,y,g,\mathcal T)\), where n is remaining horizon, y the observed follower mode, g the context and \(\mathcal T\) the remaining feasible types. Deterministic threshold observations eliminate inconsistent types. The posterior is the original prior renormalized on \(\mathcal T\). It does not reveal the realized type.

For action a, enumerate next context \(g'\) and next observed follower mode \(y'\). The known update partitions the feasible types. Bellman recursion gives

\[
Q_n(h,a)=r(a,y,g)+\gamma\sum_{g',y'}P(g',y'\mid h,a)
             \max_{a'}Q_{n-1}(h',a'),\qquad Q_0=0.
\]

The recursion is exact for the finite horizon, declared type law and observation set. It uses future observations through their probabilities rather than disclosing their realizations. The benchmark solves one player's decision problem against the specified follower. Its optimality is conditional on that response law. Computing an equilibrium of two unrestricted forward-looking policies is a further problem.

## The equal-means counterexample

Take horizon 8, discount .97, S=.46, c=.45, public-context flip probability .15 and memory rate .5. Start in memory 0, mode 0 and context 1. Consider two public priors in the support order given above:

\[
p^+=(.105,.295,.390,.010,.200),\quad
p^-=(.295,.105,.010,.390,.200).
\]

Both have exactly the same mean release threshold 1.8 and mean activation threshold 4.2 in grid units. The prior difference annihilates normalization and the two threshold moments. The observation and current payoff are also identical.

Exact rational enumeration, with optimal continuation after each candidate first action, gives

\[
\frac{Q^{+}(1)-Q^{+}(0)}{Z_8}=.010997234332965\ldots,\qquad
\frac{Q^{-}(1)-Q^{-}(0)}{Z_8}=-.018100328536930\ldots,
\]
where \(Z_8=\sum_{t=0}^7.97^t\).

Thus the optimal first action depends on features of the prior beyond its two mean thresholds. The exact rational numerators and denominators are retained in exact_witness.json. The script uses rational arithmetic throughout and constructs transitions independently of the fitted model or floating-point solver.

Consider any first-action rule whose information is compressed to those two means and the same current public state. It must use the same probability w of action 1 in both cases. Even granting the exact best continuation afterward, its normalized regret is at least \(a(1-w)\) in the plus case and \(bw\) in the minus case, where \(a=.010997\ldots\), \(b=.018100\ldots\). Minimizing the larger loss gives

\[
\inf_w\max\{a(1-w),bw\}=\frac{ab}{a+b}
=.0068409012574115\ldots.
\]

The lower bound is **0.68409 payoff points**. It applies to the stated first-decision compression and pair of priors. A policy receiving the full prior, extra informative measurements, or a different history is outside that restriction. The example was found through exploratory search on development configurations and subsequently checked with exact arithmetic. It is a verified counterexample rather than a held-out statistical estimate.

## An exact measure of how close a learned policy comes

For any learned policy \(\pi\), define its Bellman regret at h as \(V^*(h)-Q^*(h,\pi(h))\). In this declared finite model,

\[
V^*(h_0)-V^\pi(h_0)=
\mathbb E_\pi\left[\sum_{t=0}^{H-1}\gamma^t
       \big(V^*(H_t)-Q^*(H_t,\pi(H_t))\big)\right].
\]

The equality follows by adding and subtracting the optimal next-state values and telescoping to terminal value zero. Both sides can be calculated exactly here. It measures accumulated decision loss under the learned policy's actual distribution of histories. Agreement with the teacher's actions on a sampled dataset is a separate diagnostic and need not imply a small payoff gap.

## Contribution and priority

Partial observation already makes the full belief relevant to optimal decisions. The new calculation specializes that principle to a compact coordination game with hidden activation and release thresholds and quantifies a precise failure of mean compression. It is a candidate component for the proposed thesis. Establishing a novel general theorem requires comparing it with belief-based planning, learning to steer opponents, and existing switching theory. The empirical study tests whether the explicit response information helps a trained policy close a measurable gap. [Ross et al., JMLR](https://jmlr.csail.mit.edu/papers/v12/ross11a.html) · [Zhang, Ma and Mazumdar, ICML](https://proceedings.mlr.press/v267/zhang25bd.html) · [Catsis, Hall and Jeffrey, Physica D](https://doi.org/10.1016/j.physd.2025.134679)
