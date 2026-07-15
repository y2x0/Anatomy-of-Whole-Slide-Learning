# Foundation Model Failure Calculus

This note develops a task-relative failure calculus for pathology foundation
models. It does not repeat the paper-by-paper tests in the
[paper-specific failure matrix](../paper_specific_derivations/11_paper_specific_failure_matrix.md).
Its purpose is narrower: to distinguish information destruction from head
restriction, training-objective non-identifiability, nuisance sensitivity, and
finite-sample failure.

All learned parameters are treated as fixed unless training randomness is
explicitly introduced. Conditional distributions are assumed to admit regular
versions. Entropy differences are used only when finite, so no expression
silently subtracts two infinite quantities. Logarithms are natural logarithms.

## 1. Failure Is Relative To A Task

Let the primary pathology observation be:

```math
X
\in
\mathcal X,
```

the target be:

```math
Y
\in
\mathcal Y,
```

and the exported representation be:

```math
Z
=
f_{\phi}(X).
```

For loss `\ell` and decision class `\mathcal H`, define the best achievable
risk from information object `U`:

```math
\mathcal R_{\ell,\mathcal H}^{\star}(U)
=
\inf_{h\in\mathcal H}
\mathbb E
\left[
    \ell
    \left(
        Y,
        h(U)
    \right)
\right].
```

A protocol-relative transfer gap is:

```math
\Delta_{\ell,\mathcal H,\mathcal H_X}(Z;X)
=
\mathcal R_{\ell,\mathcal H}^{\star}(Z)
-
\mathcal R_{\ell,\mathcal H_X}^{\star}(X).
```

This quantity depends on two decision classes. It need not be nonnegative when
the classes differ. It therefore cannot, by itself, prove that the
representation destroyed information.

Three distinct statements are often collapsed into one reported performance
drop:

```math
\begin{aligned}
\text{representation failure}
&:
\text{task information is absent from }Z,\\
\text{head approximation failure}
&:
\text{task information is present but inaccessible to }\mathcal H,\\
\text{fitted-head failure}
&:
\text{the trained rule does not attain the best risk in }\mathcal H.
\end{aligned}
```

The first is an information claim. The second is a function-class claim. The
third depends on sample size, regularization, optimization, and calibration.

## 2. Bayes Failure Removes The Head Confound

Let `\mathcal H_{\mathrm{all}}` contain every measurable decision rule for
which the risk exists. Define Bayes risk:

```math
\mathcal R_{\ell}^{\star}(U)
=
\inf_{h\in\mathcal H_{\mathrm{all}}}
\mathbb E
\left[
    \ell
    \left(
        Y,
        h(U)
    \right)
\right].
```

The irreducible representation gap is:

```math
\Delta_{\ell}^{\mathrm{Bayes}}(Z;X)
=
\mathcal R_{\ell}^{\star}(Z)
-
\mathcal R_{\ell}^{\star}(X).
```

Because `Z` is a measurable function of `X`, every rule based on `Z` can be
implemented from `X`. Therefore:

```math
\boxed{
\Delta_{\ell}^{\mathrm{Bayes}}(Z;X)
\geq
0.
}
```

A strictly positive Bayes gap proves that no replacement linear probe,
attention head, prompt set, prototype bank, or decoder can recover the missing
task information from `Z` alone.

Bayes sufficiency for the task and loss means:

```math
\mathcal R_{\ell}^{\star}(Z)
=
\mathcal R_{\ell}^{\star}(X).
```

This is weaker than invertibility of `f_{\phi}`. A representation may discard
most of the input while preserving everything needed for one target.

## 3. Exact Bayes Gaps

### 3.1 Log Loss

Assume a discrete target and unrestricted probabilistic prediction under log
loss. Bayes risk from `U` is conditional entropy:

```math
\mathcal R_{\log}^{\star}(U)
=
H(Y\mid U).
```

For deterministic `Z=f_{\phi}(X)`:

```math
\begin{aligned}
\Delta_{\log}^{\mathrm{Bayes}}(Z;X)
&=
H(Y\mid Z)
-
H(Y\mid X)\\
&=
I(Y;X\mid Z).
\end{aligned}
```

Hence:

```math
\boxed{
\Delta_{\log}^{\mathrm{Bayes}}(Z;X)
=
I(Y;X\mid Z).
}
```

The representation fails exactly to the extent that the original observation
still contains target information after conditioning on the representation.

### 3.2 Square Loss

Assume a square-integrable scalar target. The Bayes rule from `U` is:

```math
m_U(U)
=
\mathbb E[Y\mid U].
```

Bayes risk is:

```math
\mathcal R_2^{\star}(U)
=
\mathbb E
\left[
    \mathrm{Var}(Y\mid U)
\right].
```

The law of total variance gives:

```math
\boxed{
\Delta_2^{\mathrm{Bayes}}(Z;X)
=
\mathbb E
\left[
    \mathrm{Var}
    \left(
        \mathbb E[Y\mid X]
        \mid
        Z
    \right)
\right].
}
```

The lost object is not arbitrary input variation. It is variation of the
full-information conditional mean inside a representation fiber.

### 3.3 Loss Dependence

Sufficiency is loss dependent. Under log loss, the complete posterior is the
relevant statistic. Under square loss, only the conditional mean is required:

```math
P(Y\in\cdot\mid X)
\quad\text{versus}\quad
\mathbb E[Y\mid X].
```

Two encoders can therefore be equivalent for mean regression and inequivalent
for calibrated probabilistic prediction.

## 4. Fibers Turn Failure Into A Collision Problem

Define the representation fiber through `x`:

```math
\mathcal F_f(x)
=
\left\{
    x'
    \in
    \mathcal X
    :
    f_{\phi}(x')
    =
    f_{\phi}(x)
\right\}.
```

For log loss, exact task sufficiency means that there exists a measurable
posterior map `g` such that:

```math
P(Y\in\cdot\mid X)
=
g(f_{\phi}(X))
\qquad
P_X\text{-a.s.}
```

Equivalently, posterior constancy holds within the conditional fiber measure:

```math
P(Y\in\cdot\mid X=x)
=
P(Y\in\cdot\mid Z=z)
```

for `P_{X\mid Z=z}`-almost every `x` and `P_Z`-almost every
`z`. Pointwise equality for every pair in a continuous-space fiber is
stronger than the almost-sure sufficiency statement and is not required.

### 4.1 Exact Two-Point Cost

Let two observations occur with equal probability and have deterministic,
opposite labels:

```math
P(X=x_0)
=
P(X=x_1)
=
\frac{1}{2},
```

```math
Y(x_0)
=
0,
\qquad
Y(x_1)
=
1.
```

If:

```math
f_{\phi}(x_0)
=
f_{\phi}(x_1),
```

then:

```math
H(Y\mid X)
=
0,
\qquad
H(Y\mid Z)
=
\log 2.
```

Therefore:

```math
\boxed{
\Delta_{\log}^{\mathrm{Bayes}}(Z;X)
=
\log 2.
}
```

This is an exact irreducible cost in nats, not an optimization anecdote.

### 4.2 Approximate Collision Bound

Let the target score `s^{\star}` separate two observations by margin:

```math
\left|
    s^{\star}(x_0)
    -
    s^{\star}(x_1)
\right|
\geq
\gamma.
```

Suppose their embeddings are close:

```math
\left\|
    f_{\phi}(x_0)
    -
    f_{\phi}(x_1)
\right\|
\leq
\epsilon.
```

For any `L`-Lipschitz downstream score `h`:

```math
\left|
    h(f_{\phi}(x_0))
    -
    h(f_{\phi}(x_1))
\right|
\leq
L\epsilon.
```

The triangle inequality implies:

```math
\boxed{
\max_{r\in\{0,1\}}
\left|
    h(f_{\phi}(x_r))
    -
    s^{\star}(x_r)
\right|
\geq
\max
\left\{
    0,
    \frac{\gamma-L\epsilon}{2}
\right\}.
}
```

Approximate invariance is harmless only when the induced contraction is small
relative to the task margin and the permitted reader sensitivity.

## 5. Training Contracts Are Not Deployment Chains

This distinction is essential. A crop, mask, stain perturbation, or paired text
object may be used to train parameters without being applied at inference.

Let the training sample and optimizer randomness be:

```math
\mathcal D_{\mathrm{pre}},
\qquad
\Xi,
```

and let the learned parameters be:

```math
\widehat\phi
=
\mathcal A
\left(
    \mathcal D_{\mathrm{pre}},
    \Xi;
    \mathcal K_{\mathrm{view}},
    \mathcal S_{\mathrm{pre}},
    \mathcal L_{\mathrm{pre}}
\right).
```

Here:

```math
\begin{aligned}
\mathcal K_{\mathrm{view}}
&=
\text{training view, mask, crop, or corruption law},\\
\mathcal S_{\mathrm{pre}}
&=
\text{positive, negative, teacher, text, or knowledge relation},\\
\mathcal L_{\mathrm{pre}}
&=
\text{pretraining loss}.
\end{aligned}
```

These objects shape `\widehat\phi`. They are not automatically nodes in the
deployed inference chain.

For example, [Prov-GigaPath](https://www.nature.com/articles/s41586-024-07441-w)
uses masked slide-token prediction during pretraining, but the downstream
slide encoder receives a deployment tile sequence. The training mask defines
an identifiability pressure; it does not itself prove inference-time loss of
the hidden tiles.

### 5.1 Training Non-Identifiability

Define the population pretraining objective:

```math
\mathcal L_{\mathrm{pre}}(\phi)
=
\mathbb E
\left[
    \ell_{\mathrm{pre}}
    \left(
        \phi;
        V,
        S_{\mathrm{pre}}
    \right)
\right].
```

Let its minimizer set be nonempty:

```math
\Phi^{\star}
=
\arg\min_{\phi}
\mathcal L_{\mathrm{pre}}(\phi).
```

For downstream target `Y` and loss `\ell`, objective success identifies
loss-relative task sufficiency only if every admissible minimizer is
sufficient:

```math
\forall
\phi
\in
\Phi^{\star},
\qquad
\Delta_{\ell}^{\mathrm{Bayes}}
\left(
    f_{\phi}(X);
    X
\right)
=
0.
```

Under log loss, this is equivalent to posterior sufficiency:

```math
I
\left(
    Y;
    X
    \mid
    f_{\phi}(X)
\right)
=
0.
```

If two minimizers attain the same pretraining risk but differ in downstream
Bayes risk, the objective does not identify the desired task representation:

```math
\exists
\phi_1,\phi_2
\in
\Phi^{\star}
:
\quad
\mathcal R_{\ell}^{\star}(f_{\phi_1}(X))
\neq
\mathcal R_{\ell}^{\star}(f_{\phi_2}(X)).
```

This is an objective-level ambiguity. It is different from an actual
deployment collision for one trained encoder.

## 6. A Correct Deployment Information Telescope

Condition on the realized training data, optimizer randomness, fitted
parameters, and all fixed preprocessing choices. They are suppressed from the
notation below.

Let patch or tile encoding be `E`. Let `G_C` be context geometry and `G_R` be
readout geometry. Geometry may include coordinates, adjacency, ordering,
regions, or a metric relation. Let `\mathcal M_R` be the complete fixed
memory consumed by `R`, including archive embeddings, labels, prototypes,
text, and metadata when retrieval is used. For a non-retrieval system, let
`\mathcal M_R` be a fixed null object. Start from the complete deployment
information object:

```math
A_0
=
\left(
    X,
    G_C,
    G_R,
    \mathcal M_R
\right).
```

Define a deterministic chain that carries every side-information object until
the stage that consumes it:

```math
\begin{aligned}
A_1
&=
\left(
    E(X),
    G_C,
    G_R,
    \mathcal M_R
\right),\\
A_2
&=
\left(
    C(E(X);G_C),
    G_R,
    \mathcal M_R
\right),\\
A_3
&=
R
\left(
    C(E(X);G_C);
    G_R,
    \mathcal M_R
\right),\\
A_4
&=
H(A_3).
\end{aligned}
```

Now every stage is a measurable function of the preceding stage:

```math
Y
\longrightarrow
A_k
\longrightarrow
A_{k+1}.
```

Under finite log-loss Bayes risks:

```math
H(Y\mid A_4)
-
H(Y\mid A_0)
=
\sum_{k=0}^{3}
I(Y;A_k\mid A_{k+1}).
```

Define:

```math
\begin{aligned}
\Delta_E^{\mathrm{info}}
&=
I(Y;A_0\mid A_1),\\
\Delta_C^{\mathrm{info}}
&=
I(Y;A_1\mid A_2),\\
\Delta_R^{\mathrm{info}}
&=
I(Y;A_2\mid A_3),\\
\Delta_H^{\mathrm{info}}
&=
I(Y;A_3\mid A_4).
\end{aligned}
```

Then:

```math
\boxed{
H(Y\mid A_4)
-
H(Y\mid A_0)
=
\Delta_E^{\mathrm{info}}
+
\Delta_C^{\mathrm{info}}
+
\Delta_R^{\mathrm{info}}
+
\Delta_H^{\mathrm{info}}.
}
```

This is an exact information decomposition. It does not decompose the risk of
a fitted head into approximation, estimation, optimization, and calibration.
The last term measures information discarded by replacing the head input with
the deterministic head output.

If geometry and memory are deterministic functions of `X`, including them in
`A_0` changes no information. If either is external side information, it
must remain in the state. Writing `C(E(X);G_C)` as though it were a
garbling of `E(X)` alone would be false whenever `G_C` contributes
target information. Likewise, retrieval cannot be represented by a metric
relation alone when the readout consumes archive values. Bayes gaps written
against `X` elsewhere in this note use the primary pathology observation as
reference; complete-system gaps use `A_0`.

## 7. Restricted Heads Have A Different Decomposition

Let `Z_s=A_3` be the slide statistic before the task head. For a declared head
class `\mathcal H`, define the approximation gap:

```math
\Delta_H^{\mathrm{approx}}
=
\mathcal R_{\ell,\mathcal H}^{\star}(Z_s)
-
\mathcal R_{\ell}^{\star}(Z_s)
\geq
0.
```

For a fitted rule `\widehat h\in\mathcal H`, define fitted excess risk:

```math
\Delta_H^{\mathrm{fit}}
=
\mathbb E
\left[
    \ell
    \left(
        Y,
        \widehat h(Z_s)
    \right)
\right]
-
\mathcal R_{\ell,\mathcal H}^{\star}(Z_s)
\geq
0.
```

The total excess over full-observation Bayes risk is:

```math
\begin{aligned}
&
\mathbb E
\left[
    \ell
    \left(
        Y,
        \widehat h(Z_s)
    \right)
\right]
-
\mathcal R_{\ell}^{\star}(A_0)\\
&=
\underbrace{
\mathcal R_{\ell}^{\star}(Z_s)
-
\mathcal R_{\ell}^{\star}(A_0)
}_{
\Delta_{ECR}^{\mathrm{Bayes}}
}
+
\Delta_H^{\mathrm{approx}}
+
\Delta_H^{\mathrm{fit}}.
\end{aligned}
```

Only the first term proves information loss before the head. The second can be
changed by enlarging the decision class. The third can change with training
data, regularization, optimization, or calibration.

There is no universal nonnegative split of `\Delta_H^{\mathrm{fit}}` into
estimation and optimization components without specifying an empirical
objective, algorithm, and comparison rule. Those labels should not be added
casually.

## 8. Sufficiency And Robustness Are Different Axes

An encoder can be task sufficient in one environment and fragile under a
nuisance intervention. Conversely, it can be nuisance invariant while deleting
task signal.

Let morphology be `B`, nuisance be `A`, and exogenous variables be `U_X` and
`U_Y`. Assume the target has no causal dependence on the nuisance under the
declared intervention:

```math
Y
=
r(B,U_Y),
```

and assume the structural image mechanism:

```math
X
=
g(B,A,U_X).
```

For a declared intervention from `a` to `a'`, couple the counterfactual images
by holding `B`, `U_X`, `U_Y`, and therefore `Y` fixed:

```math
X^{(a)}
=
g(B,a,U_X),
\qquad
X^{(a')}
=
g(B,a',U_X).
```

For a fixed deployed predictor `q`, define the signed interventional risk
effect:

```math
\Delta_{a\rightarrow a'}^{\mathrm{int}}
=
\mathbb E
\left[
    \ell
    \left(
        Y,
        q(X^{(a')})
    \right)
    -
    \ell
    \left(
        Y,
        q(X^{(a)})
    \right)
\right].
```

Positive values indicate average degradation under the declared intervention;
negative values indicate average improvement. Cancellation can hide strong
subgroup effects, so also consider:

```math
\mathbb E
\left[
    \left|
        \ell(Y,q(X^{(a')}))
        -
        \ell(Y,q(X^{(a)}))
    \right|
\right]
```

and subgroup-conditional effects:

```math
\Delta_{a\rightarrow a'}^{\mathrm{int}}(b)
=
\mathbb E
\left[
    \ell(Y,q(X^{(a')}))
    -
    \ell(Y,q(X^{(a)}))
    \mid
    B=b
\right].
```

Stain normalization or scanner simulation identifies this causal quantity only
when it is a valid intervention under the structural model. A site-held-out
split is an environment-shift test, not automatically an estimator of a
single-variable intervention.

The two main axes are therefore:

```math
\begin{aligned}
\text{log-loss Bayes task insufficiency}
&=
I(Y;X\mid Z),\\
\text{interventional sensitivity}
&=
\Delta_{a\rightarrow a'}^{\mathrm{int}}.
\end{aligned}
```

Neither determines the other.

## 9. Objective-Specific Failure Corollaries

The results below are derived diagnostics. The cited papers instantiate the
objective structures; the papers do not necessarily state these corollaries.

### 9.1 Contrastive False Repulsion

First consider the generic one-positive lemma. Let anchor `i` have one declared
positive `i^+` and negative index set `\mathcal B_i`. Define:

```math
\mathcal J_i
=
\{i^+\}
\cup
\mathcal B_i,
\qquad
\tau
>
0.
```

For similarity logits `s_{ij}`:

```math
p_{ij}
=
\frac{
    \exp(s_{ij}/\tau)
}{
    \sum_{k\in\mathcal J_i}
    \exp(s_{ik}/\tau)
}.
```

The one-positive InfoNCE loss is:

```math
\mathcal L_i
=
-\log p_{ii^+}.
```

For a declared negative `j` distinct from `i^+`:

```math
\boxed{
\frac{
    \partial\mathcal L_i
}{
    \partial s_{ij}
}
=
\frac{p_{ij}}{\tau}.
}
```

If `j` is semantically compatible with `i`, this is direct false-repulsion
pressure. The total direct false-negative logit mass is:

```math
M_i^{\mathrm{FN}}
=
\sum_{
    j\in\mathcal B_i
    :
    \rho(i,j)=1
}
p_{ij},
```

where `\rho` is an independently specified semantic-compatibility relation.
The direct logit-gradient mass is:

```math
\frac{M_i^{\mathrm{FN}}}{\tau}.
```

This does not equal parameter-space damage. The anchor-local parameter gradient
also contains similarity Jacobians:

```math
\nabla_{\phi}\mathcal L_i
=
\sum_{j\in\mathcal J_i}
\frac{
    \partial\mathcal L_i
}{
    \partial s_{ij}
}
\nabla_{\phi}s_{ij}.
```

Cross-anchor contributions appear in the batch gradient:

```math
\nabla_{\phi}
\sum_r
\mathcal L_r
=
\sum_r
\sum_{j\in\mathcal J_r}
\frac{
    \partial\mathcal L_r
}{
    \partial s_{rj}
}
\nabla_{\phi}s_{rj}.
```

The correct claim is therefore local: large false-negative softmax mass may
dominate the direct gradient mass among negative coordinates, not the full
logit-gradient norm or parameter update. The positive-coordinate magnitude is:

```math
\frac{
    1-p_{ii^+}
}{
    \tau
},
```

which equals the total negative-coordinate gradient mass.

#### CTransPath Specialization

[CTransPath](https://pubmed.ncbi.nlm.nih.gov/35952419/) does not use the
one-positive numerator above after its warmup. For anchor `z`, it uses mined
positive set `A` and current-minibatch negative set `N`:

```math
\mathcal L_{\mathrm{SRCL}}(z;A,N)
=
-\log
\frac{
    \displaystyle
    \sum_{a\in A}
    \exp(a^{\mathsf T}z/\tau)
}{
    \displaystyle
    \sum_{a\in A}
    \exp(a^{\mathsf T}z/\tau)
    +
    \sum_{n\in N}
    \exp(n^{\mathsf T}z/\tau)
}.
```

Conditional on the discrete mined support remaining fixed:

```math
\boxed{
\frac{
    \partial\mathcal L_{\mathrm{SRCL}}
}{
    \partial z
}
=
-
\frac{
    \sum_{a\in A}
    \exp(a^{\mathsf T}z/\tau)a
}{
    \tau
    \sum_{a\in A}
    \exp(a^{\mathsf T}z/\tau)
}
+
\frac{
    \sum_{w\in A\cup N}
    \exp(w^{\mathsf T}z/\tau)w
}{
    \tau
    \sum_{w\in A\cup N}
    \exp(w^{\mathsf T}z/\tau)
}.
}
```

A false mined positive changes both the positive-set barycenter and the
all-candidate barycenter. It creates false attraction rather than the false
repulsion described by the generic negative derivative. The top-support map is
discrete, so this derivative is conditional on the selected support.

#### RetCCL Specialization

[RetCCL](https://pubmed.ncbi.nlm.nih.gov/36270093/) weights selected negative
logits before exponentiation. For anchor `a`, positive key `k`, negative `v`,
and cluster-dependent weight `\varphi(v)`:

```math
\ell_{\mathrm{W\mbox{-}NCE}}(a,k)
=
-\log
\frac{
    \exp(a^{\mathsf T}k/\tau)
}{
    \exp(a^{\mathsf T}k/\tau)
    +
    \sum_v
    \exp
    \left(
        \varphi(v)a^{\mathsf T}v/\tau
    \right)
}.
```

If `\pi_v` is the resulting denominator probability and
`s_v=a^{\mathsf T}v`, then, conditional on fixed cluster assignment and
fixed `\varphi(v)`:

```math
\boxed{
\frac{
    \partial\ell_{\mathrm{W\mbox{-}NCE}}
}{
    \partial s_v
}
=
\frac{
    \varphi(v)\pi_v
}{
    \tau
}.
}
```

Thus RetCCL scales the local derivative to `\varphi(v)\pi_v/\tau`; it
does not simply delete denominator mass. This formula alone does not guarantee
attenuation relative to unweighted InfoNCE because `\pi_v` also changes
with the weighted logit. Its separate centroid-level contrastive objective
requires a second analysis and is not represented by this derivative.

### 9.2 Masked Prediction Identifies A Conditional Target

Let `M` be a random mask, `T_M` the hidden target, and `X_{\bar M}` the visible
context. For mask `m`, let the target space be `\mathcal T_m` and define
the fixed measurable codomain:

```math
\mathfrak Q
=
\bigsqcup_{m\in\mathcal M}
\mathcal P(\mathcal T_m).
```

Under log loss or another strictly proper distributional score, the
Bayes-optimal prediction object is the conditional kernel in `\mathfrak Q`:

```math
Q_M(x_{\mathrm{vis}},m)
=
P
\left(
    T_M
    \in
    \cdot
    \mid
    X_{\bar M}=x_{\mathrm{vis}},
    M=m
\right).
```

The objective constrains prediction of `Q_M`; it does not identify a unique
internal representation. Let the masked visible object be:

```math
W
=
\left(
    X_{\bar M},
    M
\right).
```

If a state preserves only this kernel, its log-loss compression gap relative
to `W` is:

```math
I
\left(
    Y;
    X_{\bar M},M
    \mid
    Q_M(X_{\bar M},M)
\right).
```

This quantity can be positive even at Bayes-optimal masked-prediction risk. It
is not the gap relative to the original unmasked slide; that comparison also
contains information removed by the stochastic masking channel.

That general result does not describe every pathology masked objective.
[Prov-GigaPath](https://www.nature.com/articles/s41586-024-07441-w) reconstructs
masked tile embeddings with squared Euclidean loss. Let hidden tile target be
`H_M`. Its Bayes target is the conditional mean:

```math
\mu_M(X_{\bar M},M)
=
\mathbb E
\left[
    H_M
    \mid
    X_{\bar M},M
\right].
```

The minimum squared reconstruction risk is:

```math
\inf_g
\mathbb E
\left[
    \left\|
        H_M
        -
        g(X_{\bar M},M)
    \right\|_2^2
\right]
=
\mathbb E
\left[
    \left\|
        H_M
        -
        \mu_M(X_{\bar M},M)
    \right\|_2^2
\right].
```

Consequently, a state from which the decoder recovers `\mu_M` is sufficient
for this reconstruction loss; it need not preserve the entire conditional law
of `H_M`. If only the conditional mean survived, its log-loss compression
gap relative to `W` would be:

```math
I
\left(
    Y;
    X_{\bar M},M
    \mid
    \mu_M(X_{\bar M},M)
\right).
```

Prov-GigaPath does not export only this conditional mean at deployment. The
formula characterizes what its squared masked-reconstruction term alone
identifies.

Suppose the visible context decomposes as:

```math
X_{\bar M}
=
(A,B),
```

where morphology `B` is task relevant and nuisance `A` is easy to reconstruct
from. If:

```math
T_M
\perp
B
\mid
A,M.
```

Then:

```math
P(T_M\in\cdot\mid X_{\bar M},M)
=
P(T_M\in\cdot\mid A,M).
```

A representation retaining `A`, while the decoder also receives `M`, may be
sufficient for the masked objective even if it discards `B`. This is a
conditional-sufficiency shortcut, not proof that masked pretraining always
learns nuisance.

The exact targets must stay separate: HIPT uses multiview DINO agreement;
UNI and Virchow use DINOv2 recipes with teacher agreement and masked-token
components; TITAN uses iBOT on feature grids; Prov-GigaPath uses squared
reconstruction of masked slide tile embeddings.

### 9.3 Teacher Targets Identify Teacher Fibers

[HIPT](https://arxiv.org/abs/2206.02647),
[UNI](https://pmc.ncbi.nlm.nih.gov/articles/PMC11403354/), and
[Virchow](https://arxiv.org/abs/2309.07778) use DINO-family teacher-student
contracts. The full teacher-statistic derivation is in
[invariance and surviving statistics](02_invariance_and_surviving_statistics.md).
Its failure corollary is:

```math
I
\left(
    Y;
    X,V
    \mid
    T_{\bar\phi}(X,V)
\right)
>
0.
```

When this holds, the teacher target alone is not task sufficient. Matching it
does not force every student to discard the residual signal because other
losses and unconstrained backbone state can preserve more information.

For constant momentum coefficient:

```math
0
\leq
m
<
1,
```

the exponential-moving-average teacher satisfies:

```math
\bar\phi_t
=
m^t\bar\phi_0
+
(1-m)
\sum_{r=1}^{t}
m^{t-r}\phi_r.
```

This is a convex average of parameters, not generally an ensemble average of
nonlinear predictor functions. It introduces no new labeled supervision beyond
the initialization and student trajectory. If `\bar\phi_0` contains external
pretrained information, that information persists through the first term.

### 9.4 Captions And Reports Define Different Channels

Let `O` denote the observed text object paired with an image or slide. Its
meaning is model specific:

```math
O
\in
\left\{
    \text{social-media description},
    \text{curated caption},
    \text{ROI caption},
    \text{WSI report}
\right\}.
```

[PLIP](https://pubmed.ncbi.nlm.nih.gov/37592105/) uses OpenPath image-text
pairs. [CONCH](https://arxiv.org/abs/2307.12914) combines image-caption
contrast with a separate autoregressive captioning path.
[TITAN](https://www.nature.com/articles/s41591-025-03982-3) includes ROI-caption
and WSI-report alignment. These observed channels must not be collapsed into a
single generic report variable.

For generative log loss, the Bayes text-prediction statistic is:

```math
Q_O(x)
=
P(O\in\cdot\mid X=x).
```

If only that statistic survives, the downstream task gap is:

```math
I(Y;X\mid Q_O(X)).
```

This exact statement applies to the generative proper-loss path. It does not
automatically characterize a contrastive representation. Under a declared
negative text law `P_N`, assume negatives are sampled independently from
`P_N`, the score class is unrestricted, a common dominating measure exists,
and:

```math
P(O\in\cdot\mid X=x)
\ll
P_N
```

for almost every `x`. Under the corresponding idealized
noise-contrastive model, the optimal finite score has density-ratio form:

```math
s^{\star}(x,o)
=
\log
\frac{
    p(o\mid x)
}{
    p_N(o)
}
+
c(x),
```

under the assumptions of the corresponding noise-contrastive model. Changing
the negative law changes the identified score geometry without changing the
conditional caption channel.

Thus failure can arise from at least three distinct sources:

```math
\begin{aligned}
\text{text invisibility}
&:
I(Y;X\mid Q_O(X))>0,\\
\text{training negative-law dependence}
&:
s^{\star}
=
s^{\star}[P_N],\\
\text{export mismatch}
&:
R_{\mathrm{text}}^{\mathrm{pre}}
\neq
R_{\mathrm{task}}^{\mathrm{deploy}}.
\end{aligned}
```

A deployment negative-law mismatch exists only when the downstream task itself
defines a candidate or negative text law, as in a declared zero-shot prompt
set or retrieval pool. Ordinary feature transfer has no intrinsic
`P_N^{\mathrm{deploy}}`.

### 9.5 Knowledge Targets Are Additional Supervision, Not The Whole Encoder

[KEEP](https://arxiv.org/html/2412.13126v1) uses disease-graph attributes,
semantic grouping, and knowledge-conditioned image-text relations. Let `K`
denote the sample-associated knowledge target. It may be supplied through a
disease entity, graph relation, or caption grouping; it need not be a
deterministic function of image pixels:

```math
K
\in
\mathcal K.
```

If the exported representation were restricted to a deterministic transform
of `K` alone:

```math
Z
=
u(K),
```

then its task information is bounded by the knowledge channel:

```math
I(Y;Z)
\leq
I(Y;K).
```

But KEEP also retains a visual encoder and uses knowledge to modify grouping
and alignment. The inequality cannot be used to claim that its complete
representation is knowledge-limited unless `u(K)` is shown to be the sole
surviving statistic. The valid audit question is narrower: which relations are
added, removed, or reweighted by the knowledge contract?

## 10. Slide Aggregation Creates Its Own Fibers

Patch sufficiency does not imply slide sufficiency. Let a slide be represented
before aggregation by:

```math
U
=
\left(
    F_p,
    G_R,
    \mathcal M_R
\right),
```

where `F_p` is the patch-feature matrix, `G_R` is any geometry available to
the aggregation rule, and `\mathcal M_R` is the complete retrieval memory
when one is used. Let:

```math
Z_s
=
R(U).
```

The aggregation Bayes gap under log loss is:

```math
\boxed{
\Delta_R^{\mathrm{info}}
=
I(Y;U\mid Z_s).
}
```

Any two slides in the same aggregation fiber are indistinguishable to every
downstream head:

```math
R(U_0)
=
R(U_1).
```

This covers mean pooling, a class token, a query token, a prototype histogram,
a hard tumor ratio, or any other deterministic export.

### 10.1 Linear Convex Pooling Example

The following is a hypothetical algebraic example, not a claim that CONCH or
TITAN uses this exact pooler. Let:

```math
F_p
=
\begin{bmatrix}
h_1^{\mathsf T}\\
\vdots\\
h_n^{\mathsf T}
\end{bmatrix}
\in
\mathbb R^{n\times d},
```

and let convex weights satisfy:

```math
a
\in
\mathrm{relint}(\Delta^{n-1}).
```

The slide vector is:

```math
z
=
F_p^{\mathsf T}a.
```

If a nonzero perturbation satisfies:

```math
\delta
\in
\ker(F_p^{\mathsf T})
\cap
\mathbf 1^{\perp},
```

then sufficiently small `\epsilon` gives another feasible weighting:

```math
a'
=
a
+
\epsilon\delta,
```

with:

```math
F_p^{\mathsf T}a'
=
F_p^{\mathsf T}a,
\qquad
\mathbf 1^{\mathsf T}a'
=
1.
```

A dimension lower bound is:

```math
\dim
\left(
    \ker(F_p^{\mathsf T})
    \cap
    \mathbf 1^{\perp}
\right)
\geq
n
-
\mathrm{rank}(F_p)
-
1.
```

Therefore, if:

```math
n
\geq
\mathrm{rank}(F_p)
+
2,
```

the aggregation equation alone generally does not identify patch weights. This
proves noninjectivity for one fixed feature matrix and pooled vector. It does
not prove that every alternative weight vector is realizable by one shared
attention network across the dataset.

## 11. Paper-Grounded Failure Claims

The table separates a paper-reported mechanism from the diagnostic derived in
this note.

| Objective or export | Exact paper anchor | Mathematical object actually constrained | Valid derived failure question |
|---|---|---|---|
| mined multi-positive contrast | CTransPath | conventional and retrieved positives in an SRCL numerator | does a mined false positive distort the positive barycenter or destabilize discrete support selection? |
| weighted instance and group contrast | RetCCL | cluster-weighted negative logits plus a separate centroid relation | how much direct false-negative gradient remains after weighting, and does the centroid target preserve task distinctions? |
| teacher-student agreement | HIPT; UNI; Virchow | teacher output across declared views or masks | does the teacher fiber join inputs with different downstream posteriors? |
| masked slide prediction | Prov-GigaPath | conditional mean of hidden tile embeddings under squared reconstruction | is task-relevant structure absent from that conditional mean or from the separately deployed slide export? |
| image-caption contrast | PLIP; CONCH | pair relation relative to a negative-text law | is the task invisible to the caption channel, or is performance driven by the negative law and prompts? |
| autoregressive captioning | CONCH | conditional caption-token law through the captioning token set | does the captioning export preserve information not present in the one-query retrieval export? |
| ROI-caption and WSI-report alignment | TITAN | distinct local-caption and global-report relations | which target is visible at each textual scale, and which query-token export reaches the task? |
| knowledge-conditioned grouping | KEEP | disease attributes, semantic groups, and hierarchy-aware pair relations | which independent semantic relations are corrected or corrupted by the knowledge prior? |

The exact architecture-level tests remain in the paper-specific matrix. This
table is only a map from objective to a mathematically admissible claim.

## 12. C/R/G/S/H Failure Placement

For deployment, write:

```math
\widehat Y
=
H_{\theta_H}
\circ
R_{\theta_R,G_R}
\circ
C_{\theta_C,G_C}
\circ
E_{\widehat\phi}(X).
```

The slots mean:

| Slot | Role | Failure object |
|---|---|---|
| `C` | contextualizes patch, region, or slide states | information lost when contextual states replace encoded states, conditional on carried geometry |
| `R` | aggregates or exports the slide statistic, optionally using a carried memory | aggregation-fiber or archive-value loss |
| `G` | supplies coordinates, order, adjacency, hierarchy, metric, or archive relation | omitted, corrupted, or shifted relational side information |
| `S` | supplies the training relation that shaped fitted parameters | objective non-identifiability or supervision-channel mismatch |
| `H` | maps the slide statistic to the task output | information compression, approximation gap, and fitted excess risk |

`S` is not an ordinary inference-time map. It acts through the learned
parameters:

```math
\widehat\Theta
=
\mathcal A
\left(
    \mathcal D_{\mathrm{pre}};
    S,
    \mathcal L_{\mathrm{pre}}
\right).
```

### 12.1 Context Failure

With geometry carried correctly:

```math
\Delta_C^{\mathrm{info}}
=
I(Y;A_1\mid A_2).
```

This measures information discarded by context formation. It does not blame
the context operator for target information supplied only later by an external
archive.

### 12.2 Readout Failure

Readout is aggregation or export:

```math
\Delta_R^{\mathrm{info}}
=
I(Y;A_2\mid A_3).
```

Mean pooling that erases cardinality, absolute counts, higher-order structure,
or distinctions between distributions with the same first moment belongs here.
A query token that omits a rare region or a hard ratio that discards confidence
also belongs here.

### 12.3 Geometry Failure

Geometry failure compares declared geometry objects while holding the other
maps fixed. For a geometry perturbation `\Gamma`:

```math
\Delta_G^{\mathrm{risk}}(\Gamma)
=
\mathbb E
\left[
    \ell
    \left(
        Y,
        q(X,\Gamma(G))
    \right)
    -
    \ell
    \left(
        Y,
        q(X,G)
    \right)
\right].
```

It is a perturbation effect unless `\Gamma` has a justified causal or
environmental interpretation.

### 12.4 Supervision Failure

Supervision failure occurs when at least one population-optimal pretraining
solution is task insufficient:

```math
\exists
\phi
\in
\Phi^{\star}
:
\quad
\Delta_{\ell}^{\mathrm{Bayes}}
\left(
    f_{\phi}(X);X
\right)
>
0.
```

Every minimizer can have the same positive gap. A stronger subtype is
minimizer-dependent transfer:

```math
\exists
\phi_1,\phi_2
\in
\Phi^{\star}
:
\quad
\Delta_{\ell}^{\mathrm{Bayes}}
\left(
    f_{\phi_1}(X);X
\right)
\neq
\Delta_{\ell}^{\mathrm{Bayes}}
\left(
    f_{\phi_2}(X);X
\right).
```

These cases include false contrastive relations, teacher blind spots,
report-invisible targets, and incorrect knowledge edges.

### 12.5 Head Failure

Head failure must be typed:

```math
\left(
    \Delta_H^{\mathrm{info}},
    \Delta_H^{\mathrm{approx}},
    \Delta_H^{\mathrm{fit}}
\right).
```

Calling all three a readout error would violate the framework because `R` is
the slide aggregation/export operator.

## 13. What Evidence Supports Which Claim

No single empirical score identifies the entire calculus.

| Evidence | Claim it can support | Claim it cannot support alone |
|---|---|---|
| lower probe error | better performance under one head and protocol | lower Bayes representation gap |
| stronger nonlinear than linear probe | head-class sensitivity | exact amount of inaccessible information |
| representation collision with known posterior separation | log-loss posterior insufficiency for the collided pair | population failure rate without a pair law |
| nuisance perturbation stability | robustness to the declared perturbation | causal invariance to scanner or site |
| interventional risk effect under a valid SCM | effect of the declared intervention | absence of redundant shortcut reliance |
| lower pretraining objective | better fit to the training relation | downstream sufficiency |
| paper-specific stage ablation | marginal utility under that training protocol | universal causal contribution of the stage |

A collision audit must declare a pair coupling:

```math
(X,X')
\sim
\Pi_{\mathrm{pair}}.
```

Examples include independent pairs, matched-morphology pairs, positive pairs,
or nearest-neighbor pairs. A posterior-separation diagnostic such as:

```math
P_{\Pi_{\mathrm{pair}}}
\left(
    \|Z-Z'\|
    \leq
    \epsilon,
    \quad
    \mathrm{TV}
    \left(
        \eta(X),
        \eta(X')
    \right)
    \geq
    \delta
\right)
```

is undefined without that coupling. For discrete targets:

```math
\mathrm{TV}(p,q)
=
\frac{1}{2}
\sum_{y\in\mathcal Y}
|p_y-q_y|.
```

The true posterior:

```math
\eta(x)
=
P(Y\in\cdot\mid X=x)
```

is generally unavailable. Any plug-in estimate is model-based unless repeated
labels or an identified probabilistic model justify it.

The complete experiment-level reporting checklist is already given in the
[paper-specific failure matrix](../paper_specific_derivations/11_paper_specific_failure_matrix.md).

## 14. Bottom Line

The central distinction is:

```math
\boxed{
\text{objective ambiguity}
\neq
\text{deployment information loss}
\neq
\text{head approximation error}
\neq
\text{fitted-model error}
\neq
\text{interventional fragility}.
}
```

Pretraining determines a set of representations compatible with a relation.
Deployment selects one fitted representation, composes it with context,
geometry, readout, and a task head, and exposes a new sequence of information
fibers. A valid failure claim must say which fiber, which task, which loss, and
which source of evidence it concerns.
