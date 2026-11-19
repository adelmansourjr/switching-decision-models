# Unequal duty cycles and finite-window action bounds

Adel Mansour · 11 November 2026

## Unequal-duty overlap

For duty fractions \(a,b\in[0,1]\), let \(L_{a,b}(\phi)\) be the measure of the phase interval on which both relays are high, with relative phase \(\phi\). Take \(\phi\in[0,1)\). By intersecting the first interval \([0,a)\) with the periodic copies of the second,

\[
L_{a,b}(\phi)=\sum_{k=0}^1
\left[\min(a,k+b-\phi)-\max(0,k-\phi)\right]_+ .
\]

Moving the second interval can add length at its leading edge and remove length at its trailing edge. The one-sided slopes of the overlap are therefore in \(\{-1,0,1\}\), giving circular Lipschitz constant 1. The maximum is \(\min(a,b)\) and the minimum is \(\max(0,a+b-1)\). The rise and fall each have length \(r=\min(a,b,1-a,1-b)\), with possible flat parts.

For a known shift \(d\in[0,1/2]\), the difference \(L_{a,b}(\phi+d)-L_{a,b}(\phi)\) has a sharp range of \([-\min(d,r),\min(d,r)]\) over phase. If \(d\le r\), place the interval of length \(d\) entirely on one slope. If \(r<d\le 1/2\), place it across the full slope and within the adjacent plateaus. These plateaus have total length \(1-2r\), so the latter placement exists. Reversing which slope is used realizes the opposite sign. Consequently a shift costing \(c>\min(d,r)\) can be ruled out from the duties alone, although the actual joint occupancy may be unknown. This rule is exact for fixed relative phase over full cycles. For a changing phase, the same bound holds for the difference of the two averaged overlap predictions if the intervention adds \(d\) to the *same* underlying phase path.

## A finite-window bridge to the simulator

Write the two relay phases in normalized time \(s\in[0,1]\) as \(Ns\) and \(Ns+\psi(s)\), where \(N\) is a positive integer. Assume fixed duties and an \(M\)-Lipschitz relative phase path in cycles per scoring window. Let \(P_{11}(\psi)\) be the actual joint high fraction. Partition into \(N\) equal first-relay cycles. On each, the relative phase moves by at most \(M/N\). Freezing it changes the second indicator only within \(M/N\) of its two phase boundaries, a fraction at most \(4M/N\) of that cycle. Replacing the frozen overlap by the mean of the moving overlap costs another \(M/N\). This yields

\[
\left|P_{11}(\psi)-\int_0^1L_{a,b}(\psi(s))\,ds\right|\le \min(1,5M/N).
\]

For the same phase intervention in both models, compare a true path \(\psi\) with an approximate path \(\tilde\psi\). The absolute error in the predicted gain from shifting is bounded by \(10M/N+2\int_0^1\operatorname{dist}(\psi(s)-\tilde\psi(s),\mathbb Z)\,ds\), before quadrature error. The factor two is for the two arms. The sharper value can be taken with the bound of one for each individual occupancy. A phase uncertainty band can be substituted for the path difference. The estimate can be loose near a decision boundary. It gives a sufficient certificate when the predicted gain lies further from zero than the bound. Testing a tighter certificate is a proposed thesis result.

The predictor used in the frozen simulator experiment retained calibrated duty fractions and the mean relative rate. It omitted the unknown shape of the relative phase path. The event solver evaluated the full path. The bounds above make that approximation explicit and show why a small error in predicting joint occupation does not automatically imply a small error in ranking two actions.

## Value of a noisy probe

Let \(X\) denote coarse information, \(Y\) a purchased phase reading, and \(G\) the true gain of shifting over keeping the current phase before sensor cost. The optimal value increment from buying a reading in a single-context Bayesian model is

\[
\mathcal V(X)=\mathbb E[(\mathbb E[G\mid X,Y])_+\mid X]-(\mathbb E[G\mid X])_+ .
\]

This follows by taking the better arm conditional on the information available, then averaging over the future reading. The incremental value is nonnegative by convexity. A second use of conditional Jensen gives the upper bound \(\mathbb E[G_+\mid X]-(\mathbb E[G\mid X])_+\), corresponding to perfect information about the gain. When outcomes and measurements do not alter other contexts, all sensor fees are equal and at most \(K\) can be purchased in a batch, the model-optimal batch choice selects the \(K\) largest positive values of \(\mathcal V(X)-\text{fee}\), or all positive values if fewer than \(K\) qualify. Exchanging a lower-priority measured context with a higher one proves the claim. This is a standard value-of-information argument specialized to the relay decision. Its ranking is only as good as the model of the action contrast and sensor. [Howard, Information Value Theory](https://doi.org/10.1109/TSSC.1966.300074)
