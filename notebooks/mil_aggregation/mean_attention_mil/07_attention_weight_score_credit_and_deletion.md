# Attention Weight, Score Credit, and Deletion in MIL

This note separates three quantities that are often rendered as the same WSI
heatmap:

1. the coefficient used to form an MIL representation;
2. a target-specific score attribution;
3. the change in the model output after an intervention.

They can correlate, but they are not interchangeable. The distinction matters
for ABMIL, CLAM, DSMIL, additive MIL, and DTFD-MIL because each method changes
either the context operator, the readout operator, or the target to which an
attribution is attached.

The central warning is:

```math
\text{attention weight}
\ne
\text{class evidence}
\ne
\text{deletion effect}.
```

An attention map is a map of one internal routing variable unless an additional
argument connects it to the final prediction.

## 1. Notation and target declaration

Let slide i contain n_i instances:

```math
H_i
=
\{h_{ij}\}_{j=1}^{n_i},
\qquad
h_{ij}
\in
\mathbb R^d.
```

An instance encoder or context operator may produce:

```math
u_{ij}
=
\mathcal C_j(H_i)
\in
\mathbb R^p.
```

For flat ABMIL, the context operator is often the identity:

```math
u_{ij}
=
h_{ij}.
```

Let the readout produce a slide representation:

```math
z_i
=
\mathcal R(H_i)
\in
\mathbb R^p.
```

The head maps z_i to a target q. Examples include:

```math
q
\in
\{
\ell_{ic},
\;r_i,
\;h_i(\tau),
\;\logit(\Pr(Y_i=1)),
\;\mathrm{sim}(z_i,q_{\mathrm{retrieval}})
\}.
```

The first obligation in an explanation is to declare q. A patch can have high
credit for a positive logit, low credit for a negative logit, and a different
credit for a survival risk at a particular horizon.

## 2. Classic gated ABMIL

The classic gated attention score for instance j is:

```math
s_{ij}
=
w^{\mathsf T}
\left[
\tanh(Vu_{ij})
\odot
\sigma(Uu_{ij})
\right].
```

Here V and U map the instance feature to an attention hidden dimension, w is
the attention projection, and the product is elementwise.

The normalized attention coefficient is:

```math
a_{ij}
=
\frac{
\exp(s_{ij})
}{
\sum_{\ell=1}^{n_i}
\exp(s_{i\ell})
},
\qquad
\sum_{j=1}^{n_i}a_{ij}
=
1.
```

With a value map v, the ABMIL representation is:

```math
z_i
=
\sum_{j=1}^{n_i}
a_{ij}v(u_{ij}).
```

The original gated attention construction commonly uses v(u) equal to u or a
learned linear transform of u. The attribution distinction does not depend on
that choice.

For a linear binary head:

```math
\ell_i
=
\beta^{\mathsf T}z_i
+
b.
```

Substituting the readout gives:

```math
\ell_i
=
\sum_{j=1}^{n_i}
a_{ij}
\beta^{\mathsf T}v(u_{ij})
+
b.
```

This equation resembles an additive decomposition, but it does not make
a_{ij} the contribution. The coefficient and the value-space class score are
separate factors.

## 3. Three different patch quantities

### 3.1 Routing weight

The attention weight is:

```math
\mathrm{weight}_{ij}
=
a_{ij}.
```

It describes how much of the current representation is formed from the value
associated with instance j. Its normalization is within the current bag.

### 3.2 Additive linear-head term

When the head is linear and the attention weights are held fixed, define:

```math
\mathrm{term}_{ij}
=
a_{ij}
\beta^{\mathsf T}v(u_{ij}).
```

The logit is:

```math
\ell_i
=
\sum_{j=1}^{n_i}
\mathrm{term}_{ij}
+
b.
```

The term is signed. A high attention weight can multiply a negative
class-direction score.

### 3.3 Deletion effect

Let H_i minus j denote the bag with instance j removed. Define:

```math
\Delta_{ij}^{(q)}
=
q(H_i)
-
q(H_i\setminus j).
```

The deletion effect recomputes the model on a changed input. It changes
attention normalization, context, and possibly the number of instances. It is
an intervention on the implemented model, not a coefficient read from one
forward pass.

These quantities coincide only under restrictive conditions:

```math
a_{ij}
\propto
\Delta_{ij}^{(q)}
```

requires an appropriate target, a fixed or specially controlled readout, a
specified deletion baseline, and no important interaction introduced by
renormalization or context recomputation.

## 4. Exact deletion identity for softmax attention

Define the unnormalized attention weight:

```math
w_{ij}
=
\exp(s_{ij}),
```

and the normalizer:

```math
S_i
=
\sum_{\ell=1}^{n_i}w_{i\ell}.
```

For compactness write:

```math
v_{ij}
=
v(u_{ij}),
\qquad
A_i
=
\sum_{\ell=1}^{n_i}
w_{i\ell}v_{i\ell}.
```

Then:

```math
z_i
=
\frac{A_i}{S_i}.
```

After deleting j and recomputing the softmax:

```math
z_{i,-j}
=
\frac{
A_i-w_{ij}v_{ij}
}{
S_i-w_{ij}
}.
```

The exact representation deletion difference is:

```math
z_i-z_{i,-j}
=
\frac{
w_{ij}
}{
S_i-w_{ij}
}
\left(
v_{ij}
-
z_{i,-j}
\right).
```

Equivalently, using a_{ij}:

```math
z_i-z_{i,-j}
=
\frac{
a_{ij}
}{
1-a_{ij}
}
\left(
v_{ij}
-
z_{i,-j}
\right).
```

This identity is exact for a flat attention-weighted mean when the scores and
values of the remaining instances are unchanged except for the deleted
instance. It shows why attention weight alone cannot determine deletion:

1. a_{ij} controls the multiplicative factor;
2. v_{ij} determines the direction of the deleted value;
3. z_{i,-j} is the recomputed remaining-bag representation.

Two patches with the same attention weight can have different deletion effects
because their values point in different directions.

## 5. Linear-head deletion effect

For the linear logit:

```math
\ell_i
=
\beta^{\mathsf T}z_i+b,
```

the exact deletion effect is:

```math
\Delta_{ij}^{(\ell)}
=
\ell_i-\ell_{i,-j}
=
\beta^{\mathsf T}
\left(
z_i-z_{i,-j}
\right).
```

Substituting the softmax identity:

```math
\Delta_{ij}^{(\ell)}
=
\frac{
a_{ij}
}{
1-a_{ij}
}
\beta^{\mathsf T}
\left(
v_{ij}
-
z_{i,-j}
\right).
```

The additive term from one forward pass is:

```math
\mathrm{term}_{ij}
=
a_{ij}
\beta^{\mathsf T}v_{ij}.
```

The two expressions differ by the denominator factor and the recomputed
baseline:

```math
\Delta_{ij}^{(\ell)}
\ne
\mathrm{term}_{ij}
\quad
\text{in general}.
```

They become proportional only in special regimes, such as a fixed normalizer
and a fixed zero baseline for all other terms. Those regimes are not the
ordinary softmax deletion experiment.

## 6. Why softmax makes patches interact

The attention coefficient depends on every score:

```math
a_{ij}
=
\frac{\exp(s_{ij})}{S_i}.
```

For the softmax Jacobian:

```math
\frac{
\partial a_{ij}
}{
\partial s_{i\ell}
}
=
a_{ij}
\left(
\mathbf 1\{j=\ell\}
-
a_{i\ell}
\right).
```

Therefore increasing the score of one patch decreases the normalized weights of
other patches. The pooled representation has cross-instance derivatives:

```math
\frac{
\partial z_i
}{
\partial s_{i\ell}
}
=
a_{i\ell}
\left(
v_{i\ell}
-
z_i
\right).
```

This is a compact expression of competition. A patch can affect the slide
output both through its own value and through how it changes the mass assigned
to every other value.

If the score network also depends on the value features, then:

```math
\frac{
\partial z_i
}{
\partial u_{ij}
}
=
\sum_{\ell=1}^{n_i}
\frac{
\partial a_{i\ell}
}{
\partial u_{ij}
}
v(u_{i\ell})
+
a_{ij}
\frac{
\partial v(u_{ij})
}{
\partial u_{ij}
}.
```

The first term contains attention-mediated interactions and the second term is
the direct value path. A heatmap based only on a_{ij} ignores both the target
head and this derivative structure.

## 7. Weight, gradient, and deletion attribution

### 7.1 Attention weight map

The raw map is:

```math
W_{ij}^{\mathrm{attn}}
=
a_{ij}.
```

It is normalized within the bag and is target-independent unless the model
uses a class-specific attention branch.

### 7.2 Gradient credit

For a scalar target q, a local gradient-times-input score is:

```math
G_{ij}^{(q)}
=
\left(
\frac{
\partial q
}{
\partial u_{ij}
}
\right)^{\mathsf T}
u_{ij}.
```

For a value-space linearization:

```math
G_{ij}^{(q)}
\approx
\left(
\frac{
\partial q
}{
\partial z_i
}
\right)^{\mathsf T}
a_{ij}v(u_{ij}).
```

This is target-specific and local. It is not the same as a_{ij}.

### 7.3 Integrated gradients

Let u_i-zero be a baseline and define a path:

```math
u_{ij}(t)
=
u_{ij}^{0}
+
t
\left(
u_{ij}
-
u_{ij}^{0}
\right),
\qquad
t\in[0,1].
```

The integrated-gradient attribution is:

```math
\mathrm{IG}_{ij}^{(q)}
=
\left(
u_{ij}
-
u_{ij}^{0}
\right)^{\mathsf T}
\int_0^1
\frac{
\partial q
\left(
U_i(t)
\right)
}{
\partial u_{ij}
}
\,dt.
```

The baseline changes the question. A zero feature baseline, a mean-patch
baseline, and a mask-and-recompute baseline do not define the same attribution.

### 7.4 Deletion

Deletion is:

```math
\Delta_{ij}^{(q)}
=
q(H_i)
-
q(H_i\setminus j).
```

It is global with respect to the chosen intervention. For attention MIL it
recomputes the denominator and can change the score of every remaining patch.

### 7.5 Leave-one-out normalized deletion

For comparison across slides, one can normalize deletion by the output range:

```math
\widetilde\Delta_{ij}^{(q)}
=
\frac{
q(H_i)-q(H_i\setminus j)
}{
|q(H_i)|+\epsilon
}.
```

This is a reporting choice, not an intrinsic property of the method.

## 8. Target dependence

Suppose the model has multiple output targets:

```math
q_i^{(1)}
=
\ell_{i,\mathrm{tumor}},
\qquad
q_i^{(2)}
=
\ell_{i,\mathrm{normal}},
\qquad
q_i^{(3)}
=
r_i.
```

The same patch can receive distinct credits:

```math
G_{ij}^{(1)}
\ne
G_{ij}^{(2)}
\ne
G_{ij}^{(3)}.
```

For a survival model with a discrete hazard head, the target may be one
horizon-specific logit:

```math
q_i^{(k)}
=
\ell_i^{(k)},
```

or a cumulative risk:

```math
q_i(\tau_k)
=
1-
\prod_{r=1}^{k}
\left(
1-h_i^{(r)}
\right).
```

An attention map learned for a classification head does not automatically
explain the survival risk at horizon tau-k. The head and the target must be
named.

For a retrieval score against prototype q:

```math
q_i^{\mathrm{retrieval}}
=
\mathrm{sim}
\left(
z_i,
q
\right),
```

the gradient can emphasize a feature direction that is irrelevant to the
classification logit. Explanation is conditional on the scalar function being
explained.

## 9. When an additive decomposition is legitimate

For a linear head and a fixed attention vector:

```math
\ell_i
=
\sum_{j=1}^{n_i}
a_{ij}\beta^{\mathsf T}v_{ij}
+
b.
```

The summands form an exact decomposition of the current logit. The exactness
does not imply causal independence:

```math
\mathrm{term}_{ij}
\text{ is an additive decomposition of }
\ell_i
\quad
\text{but not necessarily }
\Delta_{ij}^{(\ell)}.
```

If the attention weights are recomputed after deleting j, the decomposition
changes. If the context operator mixes instances before attention, the value
v_ij itself depends on other patches. If the head is nonlinear, there is no
unique additive decomposition without choosing a path or game.

## 10. Shapley credit and interaction

Let N_i be the set of patches and q(S) the output when only subset S is
retained under a specified masking convention. The Shapley value for patch j is:

```math
\phi_{ij}^{(q)}
=
\sum_{S\subseteq N_i\setminus\{j\}}
\frac{
|S|!\left(n_i-|S|-1\right)!
}{
n_i!
}
\left[
q(S\cup\{j\})
-
q(S)
\right].
```

Shapley credit averages marginal contributions across all coalitions. Deletion
uses one coalition, namely the full bag without j:

```math
\Delta_{ij}^{(q)}
=
q(N_i)
-
q(N_i\setminus\{j\}).
```

The two coincide only in special additive games. In an attention model, the
normalizer and context make marginal contributions coalition-dependent.

Pairwise interaction can be defined by a second-order deletion contrast:

```math
I_{ij\ell}^{(q)}
=
q(H_i)
-
q(H_i\setminus j)
-
q(H_i\setminus\ell)
+
q(H_i\setminus\{j,\ell\}).
```

If I is nonzero, explaining only one patch's isolated deletion omits a
pairwise effect. Attention competition often creates such interactions even
when the value head is linear.

## 11. Set invariance and coordinate interventions

Flat ABMIL is permutation invariant if the instance score and value maps are
shared and the readout is a sum:

```math
\mathcal R
\left(
\{u_{\pi(j)}\}_{j=1}^{n_i}
\right)
=
\mathcal R
\left(
\{u_j\}_{j=1}^{n_i}
\right).
```

The attention map is permutation equivariant:

```math
a_{\pi(j)}
\left(
\pi H_i
\right)
=
a_j(H_i).
```

The index of a plotted heatmap is not itself the mathematical object. To
render a map, attach the coefficient or credit back to the patch coordinate:

```math
\mathrm{map}_q(c_{ij})
=
\mathrm{score}_{ij}^{(q)}.
```

Deleting a coordinate-defined region is a different intervention from deleting
one arbitrary patch:

```math
\Delta_{i,A}^{(q)}
=
q(H_i)
-
q
\left(
H_i\setminus
\{j:c_{ij}\in A\}
\right).
```

Region deletion can test whether a spatially coherent area matters. It also
changes the bag size and the attention normalizer.

## 12. Mean pooling as a control

For mean pooling:

```math
z_i^{\mathrm{mean}}
=
\frac{1}{n_i}
\sum_{j=1}^{n_i}v_{ij}.
```

Deleting j gives:

```math
z_i^{\mathrm{mean}}
-
z_{i,-j}^{\mathrm{mean}}
=
\frac{
v_{ij}
-
z_{i,-j}^{\mathrm{mean}}
}{
n_i
}.
```

There is no learned attention normalizer, but deletion still changes the
denominator n_i. The mean coefficient one over n_i is therefore not itself a
deletion effect.

For a linear head:

```math
\Delta_{ij}^{(\ell,\mathrm{mean})}
=
\frac{1}{n_i}
\beta^{\mathsf T}
\left(
v_{ij}
-
z_{i,-j}^{\mathrm{mean}}
\right).
```

The same pattern appears: coefficient times centered value, not coefficient
alone.

## 13. Max pooling as a control

For scalar instance scores r_ij:

```math
q_i^{\mathrm{max}}
=
\max_j r_{ij}.
```

If j is not a maximizer and the maximum is unique:

```math
\Delta_{ij}^{\mathrm{max}}
=
0.
```

If j is the unique maximizer:

```math
\Delta_{ij}^{\mathrm{max}}
=
\max_jr_{ij}
-
\max_{\ell\ne j}r_{i\ell}.
```

Max pooling has an explicit witness interpretation, but it is discontinuous
at ties and has no representation-level contextual averaging. ABMIL is
smoother, but its softmax competition creates distributed interactions.

## 14. CLAM class-specific attention

For class c, CLAM uses a class-specific attention branch:

```math
a_{ij}^{(c)}
=
\frac{
\exp(s_{ij}^{(c)})
}{
\sum_{\ell=1}^{n_i}
\exp(s_{i\ell}^{(c)})
}.
```

The class-specific embedding is:

```math
z_i^{(c)}
=
\sum_{j=1}^{n_i}
a_{ij}^{(c)}v_{ij}^{(c)}.
```

This makes the weight target-indexed, but the distinction remains:

```math
a_{ij}^{(c)}
\ne
\Delta_{ij}^{(\ell_c)}
\quad
\text{in general}.
```

CLAM's instance-level clustering loss creates additional target-dependent
signals through pseudo-label selection. A high branch attention coefficient
may reflect the auxiliary clustering objective as well as the final slide
head.

If the top-k and bottom-k sets are:

```math
\mathcal T_i^{(c)}
=
\mathrm{TopK}
\left(
\{a_{ij}^{(c)}\}
\right),
\qquad
\mathcal B_i^{(c)}
=
\mathrm{BottomK}
\left(
\{a_{ij}^{(c)}\}
\right),
```

then the selection itself is a discontinuous operation. A heatmap should
state whether it displays the continuous attention coefficients, the selected
instance labels, or a final-output deletion score.

## 15. DSMIL critical-instance context

DSMIL selects a critical instance using a class score:

```math
j_i^\star
=
\arg\max_j
\rho_{ij}.
```

The critical feature conditions the attention of the remaining instances:

```math
r_{ij}
=
\mathrm{sim}
\left(
q(h_{ij}),
k(h_{i,j_i^\star})
\right).
```

The final representation can contain both a critical-instance branch and a
bag-context branch:

```math
z_i^{\mathrm{DSMIL}}
=
\lambda_i z_i^{\mathrm{critical}}
+
\left(
1-\lambda_i
\right)
z_i^{\mathrm{context}}.
```

Deleting the critical patch can change every context score because the query
or key used by the second branch changes. Its deletion effect is therefore
not recoverable from the critical attention coefficient alone.

The argmax also creates a switching boundary:

```math
\rho_{ia}
>
\rho_{ib}
\longrightarrow
j_i^\star=a,
```

```math
\rho_{ia}
<
\rho_{ib}
\longrightarrow
j_i^\star=b.
```

This is a distinct failure mode from smooth softmax competition.

## 16. Additive MIL signed evidence

An additive model may define:

```math
\ell_i^{\mathrm{add}}
=
b
+
\sum_{j=1}^{n_i}
r_{ij}.
```

The summand r_ij is an exact score-space contribution by construction:

```math
\ell_i^{\mathrm{add}}
-
\ell_i^{\mathrm{add}}(H_i\setminus j)
=
r_{ij},
```

provided the per-instance scores are unchanged when j is removed. If the
instance score is recomputed from a context operator, or if the model applies
normalization over instances, even this identity needs qualification.

Additive evidence is signed:

```math
r_{ij}
>
0
\quad
\text{supports the target logit},
```

```math
r_{ij}
<
0
\quad
\text{opposes the target logit}.
```

Softmax attention weights are nonnegative and sum to one. They cannot
represent negative evidence by themselves.

## 17. DTFD two-level credit

DTFD introduces a local attention map and a second-tier pseudo-bag map:

```math
\alpha_{i,m,j}
\quad
\text{within pseudo-bag }m,
\qquad
\gamma_{i,m}
\quad
\text{over distilled features}.
```

For AFS, a simple routing product is:

```math
\mathrm{route}_{i,m,j}
=
\gamma_{i,m}\alpha_{i,m,j}.
```

This quantity is not automatically the final slide logit credit. The final
distilled feature can be class-conditioned, and the Tier-2 head can have a
different direction. A target-specific derivative must follow the chain:

```math
\frac{
\partial q_i
}{
\partial h_{i,j}
}
=
\sum_m
\frac{
\partial q_i
}{
\partial\widehat f_{i,m}
}
\frac{
\partial\widehat f_{i,m}
}{
\partial h_{i,j}
}.
```

For MaxS and MAS, the derivative through the selected index is piecewise.
Deletion is often more interpretable for the complete pipeline, but the
partition and selection must be recomputed according to the chosen intervention
protocol.

## 18. Attention collapse is not one phenomenon

Attention collapse can mean several different things:

1. one coefficient is near one;
2. all coefficients are nearly uniform;
3. one branch is class-saturated;
4. the attention map is stable but unrelated to target deletion;
5. the attention map changes drastically under a small perturbation.

Define attention entropy:

```math
\mathcal H_i
=
-
\sum_{j=1}^{n_i}
a_{ij}\log(a_{ij}).
```

Define effective support:

```math
N_{\mathrm{eff},i}
=
\frac{1}{
\sum_{j=1}^{n_i}a_{ij}^{2}
}.
```

Low entropy or low effective support indicates concentration, not correctness.
The relationship to target faithfulness must be measured separately.

One simple concentration statistic is:

```math
\mathrm{maxmass}_i
=
\max_j a_{ij}.
```

One simple deletion agreement statistic is Spearman correlation between the
attention ranking and the deletion ranking:

```math
\rho_i^{\mathrm{rank}}
=
\mathrm{Spearman}
\left(
\mathrm{rank}(a_{ij}),
\mathrm{rank}(\Delta_{ij}^{(q)})
\right).
```

High maxmass and low rank agreement is a concrete form of attention
misalignment.

## 19. Temperature and attribution

Introduce attention temperature tau:

```math
a_{ij}^{(\tau)}
=
\frac{
\exp(s_{ij}/\tau)
}{
\sum_{\ell=1}^{n_i}
\exp(s_{i\ell}/\tau)
}.
```

As tau tends to zero, the coefficient approaches a hard maximum under a
unique score maximum:

```math
\lim_{\tau\to 0}
a_{ij}^{(\tau)}
=
\mathbf 1
\left\{
j=\arg\max_{\ell}s_{i\ell}
\right\}.
```

As tau tends to infinity:

```math
\lim_{\tau\to\infty}
a_{ij}^{(\tau)}
=
\frac{1}{n_i}.
```

Changing tau changes the representation, not merely the visualization. An
attention map generated at a post-hoc temperature is not the map of the model
that produced the original prediction unless the model was evaluated with
that temperature.

## 20. Counterexample: high attention, negative evidence

Take two value vectors in one-dimensional value space:

```math
v_{i1}=1,
\qquad
v_{i2}=-10,
```

with:

```math
a_{i1}=0.9,
\qquad
a_{i2}=0.1.
```

For a positive head with beta equal to one:

```math
\ell_i-b
=
0.9(1)
+
0.1(-10)
=
-0.1.
```

Patch one has the larger attention weight, but the overall evidence is
negative and patch two contributes negative signed evidence. A plot of
attention alone cannot express that sign.

## 21. Counterexample: equal weight, unequal deletion

Let:

```math
a_{i1}
=
a_{i2}
=
0.25,
```

but:

```math
v_{i1}
=
1,
\qquad
v_{i2}
=
-1.
```

Their attention weights coincide. Their deletion differences do not, because
the centered values relative to the remaining representation differ:

```math
z_i-z_{i,-1}
=
\frac{0.25}{0.75}
\left(
1-z_{i,-1}
\right),
```

```math
z_i-z_{i,-2}
=
\frac{0.25}{0.75}
\left(
-1-z_{i,-2}
\right).
```

The representation geometry, not the coefficient alone, determines the
intervention effect.

## 22. Counterexample: same attention, different target

Let the same representation feed two heads:

```math
\ell_i^{(1)}
=
\beta_1^{\mathsf T}z_i,
\qquad
\ell_i^{(2)}
=
\beta_2^{\mathsf T}z_i.
```

If beta-one and beta-two point in different directions, then:

```math
\Delta_{ij}^{(\ell_1)}
=
\beta_1^{\mathsf T}
\left(
z_i-z_{i,-j}
\right),
```

```math
\Delta_{ij}^{(\ell_2)}
=
\beta_2^{\mathsf T}
\left(
z_i-z_{i,-j}
\right).
```

The same patch can be positive evidence for one head and negative evidence for
another. There is no target-free notion of “the” patch importance.

## 23. Deletion protocols

Deletion is underspecified unless the intervention protocol is declared.

### 23.1 Remove and renormalize

Remove the patch and recompute attention over the remaining set:

```math
q_{i,-j}^{\mathrm{renorm}}
=
q(H_i\setminus j).
```

This is the standard set deletion for attention MIL.

### 23.2 Replace with a baseline

Replace the feature with u-zero:

```math
q_{i,\mathrm{replace}(j)}
=
q
\left(
H_i
\setminus
\{u_{ij}\}
\cup
\{u_i^0\}
\right).
```

This preserves cardinality but introduces a baseline feature.

### 23.3 Mask without renormalization

Multiply the value by a mask while retaining the original score denominator:

```math
z_i^{\mathrm{mask}(j)}
=
\sum_{\ell=1}^{n_i}
a_{i\ell}
m_{i\ell}^{(j)}
v_{i\ell},
```

where the mask is zero at j and one elsewhere. This is not the same as
remove-and-renormalize.

### 23.4 Region deletion

Delete all patches in a coordinate-defined region A:

```math
\Delta_{i,A}^{(q)}
=
q(H_i)
-
q
\left(
\{h_{ij}:c_{ij}\notin A\}
\right).
```

Region deletion tests spatial coherence but can confound area size, tissue
composition, and attention renormalization.

## 24. Faithfulness metrics

Let r_i(1), ..., r_i(n_i) be a ranking of patches. Define the cumulative
deletion curve:

```math
D_i(k)
=
q(H_i)
-
q
\left(
H_i
\setminus
\{r_i(1),\ldots,r_i(k)\}
\right).
```

The area under the deletion curve is:

```math
\mathrm{AUC}_{i}^{\mathrm{del}}
=
\frac{1}{n_i}
\sum_{k=1}^{n_i}
D_i(k).
```

For a positive logit, a larger early decrease can indicate a more faithful
ranking, but the sign and target must be fixed before aggregation across
slides.

Insertion starts from a baseline set H-zero and adds patches in ranking order:

```math
I_i(k)
=
q
\left(
H_i^{0}
\cup
\{r_i(1),\ldots,r_i(k)\}
\right)
-
q(H_i^0).
```

Deletion and insertion depend on the baseline and can disagree. They test
different counterfactuals.

## 25. Calibration versus localization

A class signal p-ij or a deletion score can be useful for ranking while being
poorly calibrated as a probability:

```math
\mathrm{rank\ quality}
\ne
\mathrm{probability\ calibration}.
```

Calibration of a patch score requires labels or a validated probabilistic
model for patches. Slide-level labels alone do not identify:

```math
\Pr(Z_{ij}=1\mid h_{ij},Y_i).
```

The DTFD class softmax, CLAM branch score, and ABMIL attention coefficient can
all be useful ranking signals without being calibrated latent posteriors.

## 26. Reporting template

Every published or internal heatmap should state:

| Field | Required declaration |
| --- | --- |
| Target | class logit, probability, hazard, risk horizon, or retrieval score |
| Quantity | attention weight, gradient, integrated gradient, additive term, deletion, or Shapley value |
| Baseline | zero, mean feature, mask, removal, or another intervention |
| Normalization | within slide, across slides, min-max, z-score, or none |
| Context | flat set, graph, transformer, state-space, or hierarchy |
| Geometry | coordinate rendering, random order, region aggregation, or none |

A minimally honest caption is:

```math
\text{map value}
=
\text{specified statistic for specified target under specified intervention}.
```

Calling the image an “attention heatmap” is insufficient when the visual is
actually a deletion or gradient map.

## 27. C/R/G/S placement

| Component | Weight map | Score credit | Deletion effect |
| --- | --- | --- | --- |
| Context C | determines features and scores before readout | enters the target derivative | recomputed under the intervention |
| Readout R | attention coefficient and weighted sum | target-specific head or path | changed normalization and representation |
| Geometry G | coordinates only affect rendering unless included in C | region or graph target changes derivative | spatial deletion defines the intervention |
| Supervision S | learned indirectly from slide labels | target head defines sign and scale | no new label; evaluates counterfactual behavior |

For flat ABMIL:

```math
H_i
\xrightarrow{\mathcal C=\mathrm{identity}}
\{u_{ij}\}
\xrightarrow{\mathcal R=\mathrm{softmax\ weighted\ mean}}
z_i
\xrightarrow{\mathcal H}
q_i.
```

The attention coefficient belongs to R. The target-specific derivative depends
on both R and H. Deletion re-evaluates the full composition.

## 28. Unified failure matrix

| Quantity shown | What it measures | What it does not prove |
| --- | --- | --- |
| raw attention | routing mass in one forward pass | positive class evidence or causal importance |
| class branch attention | class-conditioned routing | calibrated patch posterior |
| gradient-times-input | local target sensitivity | robust counterfactual effect |
| integrated gradients | path-averaged target attribution | baseline-free truth |
| additive logit term | linear score decomposition | deletion under renormalization |
| deletion | output change under one intervention | unique causal mechanism |
| Shapley value | average marginal contribution under a game | biological causality |
| DTFD routing product | local-to-global routing heuristic | final slide credit |

The failure mode is not that one statistic is always wrong. The failure mode is
using one statistic as though it answered a different question.

## 29. Design implications

If the label is a sparse witness, a model may need an upper-tail statistic:

```math
\max_j
\rho_{ij}
\quad
\text{or a high-quantile functional}.
```

If the label is distributed burden, a first moment or count-sensitive statistic
may be more appropriate:

```math
\frac{1}{n_i}
\sum_j
\rho_{ij}
\quad
\text{or}
\quad
\sum_j
\rho_{ij}.
```

If the label depends on pairwise relations, the context operator must expose
those relations:

```math
u_{ij}
=
\mathcal C_j
\left(
\{h_{i\ell}\}_{\ell=1}^{n_i}
\right),
```

because a final attention coefficient over isolated features cannot recover
interactions that were never represented.

If the label is spatial, the geometry must enter before the final readout:

```math
u_{ij}
=
\mathcal C
\left(
h_{ij},
\{h_{i\ell}:c_{i\ell}\in\mathcal N(c_{ij})\}
\right).
```

Rendering a flat attention score at a coordinate does not make the model
spatial.

## 30. Bottom line

For a patch j, report the quantity explicitly:

```math
\boxed{
\begin{aligned}
\mathrm{weight}_{ij}
&=
a_{ij},
\\
\mathrm{credit}_{ij}^{(q)}
&=
\text{target-specific attribution for }q,
\\
\Delta_{ij}^{(q)}
&=
q(H_i)-q(H_i\setminus j).
\end{aligned}
}
```

For softmax ABMIL, the exact representation deletion identity is:

```math
z_i-z_{i,-j}
=
\frac{
a_{ij}
}{
1-a_{ij}
}
\left(
v_{ij}
-
z_{i,-j}
\right).
```

That identity is the shortest mathematical reason not to equate attention
weight with deletion effect. The first is a normalized routing coefficient.
The second depends on the feature direction and the recomputed remainder of
the bag. Target-specific credit adds the head on top of that representation
change.

The correct interpretation of a MIL heatmap is therefore a contract:

```math
\text{target}
+
\text{statistic}
+
\text{baseline}
+
\text{recompute rule}
+
\text{geometry}.
```

Without those five declarations, the picture may be useful diagnostically, but
its mathematical meaning is underspecified.