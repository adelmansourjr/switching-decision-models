# From a switching example to equilibrium selection

Adel Mansour · 12 November 2026

## A precise ambition

Determine when an AI can learn a low-cost path from a low-payoff Nash equilibrium to a high-payoff one, using its own actions while facing an opponent with a hidden switching threshold. Treat the target's Nash property, its reachability and its dynamic stability as three separate mathematical questions. The AI's training objective is discounted own payoff during learning and transition, including failed commitments. A target equilibrium's payoff alone is insufficient to evaluate the policy.

Nash equilibrium fixes the unilateral-deviation criterion for the underlying game. The proposed AI work asks how an agent reaches an equilibrium through interaction when the response mechanism is partly unknown. The switching boundary makes small inference errors potentially change the choice between persisting and abandoning a costly attempt. This is the connection to Jeffrey's work.

## A tractable starting model

Let both players choose 0 or 1. The leader earns \(S>0\) at (0,0), \(R>S\) at (1,1), \(-c<0\) at (1,0), and 0 at (0,1). The follower earns \(s>0\) at (0,0), \(r>s\) at (1,1), and 0 at either mismatch. Both aligned pairs are strict Nash equilibria in the stage game, and (1,1) gives both players higher payoff.

The follower estimates the probability of leader action 1 using \(b_{t+1}=(1-\lambda)b_t+\lambda a_t\), with known \(0<\lambda<1\) in the first model. It myopically chooses its next action to maximize the stage payoff under this belief, minus a switching charge \(\kappa\), with \(0<\kappa<s\). In mode 0 it changes to 1 when \(r b\ge s(1-b)+\kappa\). In mode 1 it changes to 0 when \(s(1-b)\ge r b+\kappa\). Declare the tie convention explicitly. This gives

\[
\theta_{\rm on}=\frac{s+\kappa}{r+s},\qquad
\theta_{\rm off}=\frac{s-\kappa}{r+s}.
\]

The interval between them is hysteresis. These thresholds follow from a myopic best response with switching costs. They do not assert that the follower solves the full discounted dynamic game. The stage-game Nash claim is checked independently. Under persistent action 1, belief tends to one and follower action 1 remains stable once activated.

Starting from \(b_0=0\), the earliest activation is after

\[
n(\theta_{\rm on})=
\left\lceil\frac{\log(1-\theta_{\rm on})}{\log(1-\lambda)}\right\rceil
\]

rounds of leader action 1, for the weak activation convention. Exact integer boundaries should be checked directly in numerical code. To prove minimality, note that action 1 maximizes the next belief at every state. By induction no other sequence can make \(b_t\) exceed \(1-(1-\lambda)^t\). The leader therefore pays at least \(n\) action-1 mismatch costs on any path that activates the initially low-mode opponent. A sequence with intervening zeros needs at least as many ones and also incurs additional delay. This is a count bound, while discounting and foregone aligned reward affect the payoff cost.

For the policy that chooses 1 persistently, there are \(n\) mismatch rounds followed by the target payoff. Its infinite-horizon discounted value is

\[
V_{\rm commit}=\frac{-c(1-\gamma^n)+\gamma^n R}{1-\gamma}.
\]

The policy that remains at (0,0) earns \(S/(1-\gamma)\). Thus this commitment is better precisely when

\[
\gamma^n>\frac{S+c}{R+c}.
\]

This comparison holds for the two stated policies. The exact optimum among all history-dependent policies, with a prior over unknown thresholds, requires a separate solution. For a finite horizon \(H\), replace the final geometric tail by \(\gamma^nR(1-\gamma^{H-n})/(1-\gamma)\) if \(n<H\), and handle \(n\ge H\) as all mismatch rounds. These formulas give benchmarks for testing a learned policy's persistence decisions.

The next theorem to seek concerns unknown thresholds: determine how much payoff must be lost to identify whether a worthwhile activation is possible, and construct a policy attaining a matching bound for the stated class. With positive hysteresis, observations made after activation can also identify release behavior. This links identification, irreversible-seeming commitments over a short window, and equilibrium selection. Testing bounded sensor errors or one response delay is a controlled extension.

## What is already published

A 2024 equilibrium-steering result uses payments to guide no-regret agents and establishes budget and observation constraints. A 2025 ICML paper studies steering a no-regret learner through a player's actions without knowing its payoffs, establishes a general impossibility result, and gives positive results under more specific learning rules. A 2026 ICLR paper learns agents' utilities through payments and uses that information for steering. Work on adaptive Stackelberg learning also uses hysteresis switching within an algorithm. A comparison must distinguish that algorithmic use from physical or behavioral hysteresis in the opponent's response. These precedents make a precise response class and information set essential. [Zhang et al., EC](https://www.mit.edu/~gfarina/2024/steering_ec24/steering_ec24.pdf) · [Zhang, Ma and Mazumdar, ICML](https://proceedings.mlr.press/v267/zhang25bd.html) · [Zhang et al., ICLR](https://openreview.net/pdf?id=8yRtP2n8OK) · [Yang, Poovendran and Hespanha](https://guosong-yang.github.io/resources/YPH-CDC2019.pdf)

The threshold and geometric-series calculations are starting derivations. They do not establish priority or an architectural breakthrough. The research claim to test is a sharp cost or sample-complexity boundary for learning which commitment to make under hidden hysteretic response, together with a learned policy that approaches it in the same declared game. The existing relay archive supplies motivation and verified numerics. It provides no experiment on equilibrium steering.
