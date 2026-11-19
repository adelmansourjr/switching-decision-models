# Two threshold switches can agree on every marginal and reverse a decision

Adel Mansour · 11 November 2026

## Exact construction

Consider two autonomous relays. Relay \(i\) has a continuous state \(x_i\), a binary mode \(h_i\), and velocity \(\dot x_i=c_i(1-2h_i)\), where \(c_i>0\). It switches from 0 to 1 at \(+\epsilon\) and from 1 to 0 at \(-\epsilon\). Each trajectory starts on its periodic orbit. The period is \(4\epsilon/c_i\), and the high mode occupies half of each cycle. Let \(u_i(t)=u_i(0)+c_it/(4\epsilon)\) modulo 1, with mode 1 on the first half of the phase circle.

When \(c_1=c_2\), the relative phase \(\phi=u_2-u_1\) is constant. In a full cycle the joint high fraction is the length of the overlap of two half circles. Thus

\[
p_{11}=L(\phi)=\tfrac12-\operatorname{dist}(\phi,\mathbb Z).
\]

The one-channel high fractions are both \(1/2\). On any fixed observation window the continuous states converge uniformly to zero as \(\epsilon\to 0\), while the joint high fraction can converge to any value in \([0,1/2]\). Every endpoint is physically realizable in the model by selecting the initial phase. For a partial final cycle the occupation fraction differs from the full-cycle result by at most one period divided by the observation duration.

Let a decision maker select either the fraction of time both modes are high or a fixed normalized payoff of \(1/4\). The gamble earns \(1/2\) at synchrony and 0 at half-period separation. The optimal decision reverses. A decision rule that sees only the coarse state and the two marginal occupancies, and receives no phase, joint or outcome information before acting, must use the same gamble probability \(w\) at both endpoints. Its worst-case regret is

\[
\max\{\tfrac14(1-w),\tfrac14w\}\ \ge\ \tfrac18.
\]

The bound is sharp at \(w=1/2\). It applies to the stated single decision and information restriction. A time-aligned mode pair, sufficiently precise phase measurement, or previous informative outcomes can invalidate the indistinguishability premise. More generally, if identical allowed observations leave two models with opposing payoff gaps \(a,b>0\), the same calculation yields \(ab/(a+b)\) at \(w=a/(a+b)\) when the gamble wins in the \(a\)-gap model.

## A finite-window prediction to try to falsify

Let \(T\) be the observation duration, \(N=c_1 T/(4\epsilon)\) the number of first-relay cycles in the window, and \(\Delta=(c_2-c_1)T/(4\epsilon)\) the accumulated relative phase drift in cycles. When \(N\) grows and \(\Delta\) stays bounded, the joint-high fraction satisfies

\[
p_{11}(T)=\int_0^1 L(\phi_0+\Delta s)\,ds+E_N,
\qquad |E_N|\le \min\{1,(5|\Delta|+1)/N\}.
\]

To see this, write normalized time as \(s=t/T\) and partition the interval at relay 1's phase wraps, with a starting phase of zero. On a complete relay 1 cycle the relative phase changes by at most \(|\Delta|/N\). Freeze it at its starting value. The second mode can disagree with the frozen mode only within that distance of either of its two boundaries on the circle. This affects at most \(4|\Delta|/N\) of that cycle. Replacing the frozen overlap by the integral of \(L\) over the cycle adds at most \(|\Delta|/N\) because \(L\) is 1-Lipschitz. The final incomplete cycle adds at most \(1/N\). For an arbitrary starting phase, a corresponding bound has \(2/N\) in place of \(1/N\) to account for both partial cycles.

This gives a testable prediction that uses phase and elapsed drift measured independently of the outcome. For example, starting in synchrony with no drift gives \(1/2\), accumulated drift of half a cycle gives \(1/4\), and half-cycle separation with no drift gives 0. The half-cycle drift case reaches the decision boundary in the limit. Near it, finite timing error matters. At exact synchrony the limit retains phase dependence. At a fixed irrational frequency ratio, long observation in cycles instead gives the product duty fraction \(1/4\) by equidistribution on the two-torus. The bounded-drift result specifies an intermediate scaling where neither limiting description alone suffices. These are ideal-relay results, and transfer to perturbed relays requires new analysis and measurements.

## From invariants to a decision certificate

In a separate class of systems with modes \(j=1,\ldots,k\), let \(p\) be the vector of occupation fractions over a declared window and let \(v_j\) be the constant normal velocity of mode \(j\). Integrating the normal state equation gives \(Vp=(y(T)-y(0))/T\), where \(V\) has columns \(v_j\). In particular, a trajectory confined to \(\|y\|_\infty\le\epsilon\) gives \(\|Vp\|_\infty\le 2\epsilon/T\). For state-dependent velocities, add a bound on variation within the layer. If a switch resets the normal state, include the sum of the jumps in the balance. An ideal zero-displacement model has \(Ap=b\), where \(A\) stacks normalization and normal velocity and \(b\) supplies the known right-hand sides.

If \(r=A^\top\lambda\), the linear payoff \(r^\top p=\lambda^\top b\) is fixed over the feasible polytope. Conversely, if the polytope has a strictly positive point and \(r\) lies outside the row space of \(A\), some direction \(d\in\ker A\) satisfies \(r^\top d\ne0\). The two sufficiently small displacements \(p\pm\eta d\) both satisfy the algebraic constraints and give different payoffs. This converse is about the polytope. A physical counterexample must realize both occupation laws, as the two-relay construction does for its synchrony payoff. In the relay example, the payoff for the sum of high-mode indicators is fixed by the marginal occupancies and serves as a placebo.

For a finite window with \(\|Ap-b\|_\infty\le\zeta\), any decomposition \(r=A^\top\lambda+e\) yields

\[
|r^\top p-\lambda^\top b|\le \zeta\|\lambda\|_1+\|e\|_\infty.
\]

The inequality follows from the triangle inequality and \(\|p\|_1=1\). For two actions, compute bounds on the payoff contrast over the *joint* admissible set of their two occupation laws. A shared unknown clock, delay, or coupling links the two action models. If the lower contrast bound is positive, the first action wins for all models in the set. If the upper bound is negative, the second wins. If zero lies in the interval, the sign is unresolved by the certificate. An interval that spans both signs can reflect an outer approximation, so physically attainable witness models are required to establish real non-identifiability. This is a testable starting point for a policy that pays for a phase or joint measurement only when it can affect the action.

## An exact test for whether a measurement is worth buying

Use the same phase-locked relays. Action \(A\) keeps the relative phase. Action \(B\) shifts it by \(d\in[0,1/2]\) cycles and costs \(c\ge0\). Add the same base reward to both actions. The gain from \(B\) is \(J(\phi)-c\), where \(J(\phi)=L(\phi+d)-L(\phi)\). Since \(L\) is 1-Lipschitz on the circle, the sharp gain interval over an unknown shared phase is \([-d-c,d-c]\). The endpoints occur at \(\phi=0,1/2\). Thus \(A\) is certified when \(c>d\), although the individual payoffs can remain highly uncertain. At \(d=.10,c=.12\), the individual payoff intervals with base reward .20 are \([.20,.70]\) and \([.08,.58]\). Subtracting them separately gives a gain interval for \(B\) of \([-.62,.38]\). Using the shared phase gives \([-.22,-.02]\), certifying \(A\).

Now suppose the prior phase is uniform on the circle and a perfect, instantaneous measurement is available at a fixed payoff cost. The gain variable \(J\) has point masses \(1/2-d\) at each of \(\pm d\) and density 1 between them. To derive this, split the triangular overlap curve into its two slopes. A shift that stays on one slope gives a constant change of magnitude \(d\). The two intervals straddling the slope changes each have length \(d\), and \(J\) traverses \([-d,d]\) with absolute slope 2 on each. Before measurement, the expected gross gain of shifting is zero by translation invariance, so the best blind action is \(A\). The expected value of perfect phase information is

\[
\mathbb E[(J-c)_+]=
\begin{cases}
(1/2-d)(d-c)+\displaystyle\int_c^d(z-c)\,dz
=\tfrac12(d-c)(1-d-c), & 0\le c<d,\\
0, & c\ge d.
\end{cases}
\]

At \(d=.25,c=.10\), the break-even cost of observing phase is **.04875 of normalized payoff**, or 4.875 points. This value requires the uniform prior, the declared phase intervention and perfect instantaneous sensing. It provides a quantitative target for a noisy or delayed sensor experiment. Without delay or other effects of measurement, imperfect information can only reduce the value of information in this decision model.

## Status and relationship to published work

Catsis, Hall and Jeffrey already derive perturbation-insensitive occupation quantities, including weighted sums in systems with multiple switches. The row-space argument above is a direct finite-dimensional restatement for constant normal velocities. The proposed research extension concerns finite-window *action contrasts*, attainable payoff reversals, and whether targeted measurement can resolve those reversals in a new physical system at a useful cost. The ideal-relay calculation and its numerical check do not establish priority, transfer, or a hardware result. [Catsis, Hall and Jeffrey, Physica D, 2025](https://doi.org/10.1016/j.physd.2025.134679)
