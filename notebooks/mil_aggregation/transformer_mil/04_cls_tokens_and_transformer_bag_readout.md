# CLS Tokens and Transformer Bag Readout

## 1. Scope and central claim

This note studies the class token as a bag-level readout operator in
transformer-based multiple instance learning. The central question is:

> Is a CLS token just attention pooling with a different notation?

The short answer is:

- in a restricted one-layer model, a class token can implement a learned
  weighted average of patch values;
- in a standard transformer, the patch values and the class query are
  contextualized, so the final class token is a recurrent latent state rather
  than a single pooling row;
- with spatial operators such as PPEG, the readout is conditioned on an
  imposed geometry and is no longer a pure set functional;
- the final class-to-patch attention row is a routing statistic, not generally
  a complete causal or predictive credit assignment.

The paper-specific anchor is Shao et al., TransMIL: Transformer based
Correlated Multiple Instance Learning for Whole Slide Image Classification:

https://arxiv.org/abs/2106.00908

The general attention-as-set-aggregation comparison uses Lee et al., Set
Transformer:

https://arxiv.org/abs/1810.00825

The released TransMIL implementation supplies a 512-dimensional learned class
token, two transformer layers, PPEG between those layers, and a classifier
applied to the final class-token row. The equations here separate the exact
readout role from the spatial context operator.

## 2. Bag and token notation

For slide i, let the projected patch matrix be

```math
H_i
=
\begin{bmatrix}
h_{i1}^{\mathsf T}\\
\vdots\\
h_{in_i}^{\mathsf T}
\end{bmatrix}
\in\mathbb R^{n_i\times d}.
```

Let q^(0) be a learned class token shared across slides. The augmented
sequence is

```math
Z_i^{(0)}
=
\begin{bmatrix}
q^{(0)\mathsf T}\\
H_i
\end{bmatrix}
\in\mathbb R^{(n_i+1)\times d}.
```

Use index zero for the class row and indices one through n_i for patch rows.
The class row is not an observed instance. It is a learned state that must
acquire slide evidence through attention.

The final classifier receives

```math
z_i
=
q_i^{(L)}
\in\mathbb R^d,
\qquad
\widehat y_i
=
\psi(z_i),
```

where L is the number of transformer blocks and psi is a task head. For a
binary task, psi may be a scalar logit. For a C-class task, it may be a vector
in R^C. The class token is therefore both a representation and the input to
the final readout head.

## 3. One attention block in full

### 3.1 Multihead projections

For one head with key dimension d_h, define

```math
Q_i
=
Z_iW_Q,
\qquad
K_i
=
Z_iW_K,
\qquad
V_i
=
Z_iW_V,
```

with

```math
W_Q,W_K,W_V\in\mathbb R^{d\times d_h},
\qquad
Q_i,K_i,V_i\in\mathbb R^{(n_i+1)\times d_h}.
```

The scaled logit matrix and row-normalized attention matrix are

```math
\Lambda_i
=
\frac{Q_iK_i^{\mathsf T}}{\sqrt{d_h}},
\qquad
A_i
=
\mathrm{softmax}_{\mathrm{row}}(\Lambda_i).
```

For row r, the normalized coefficient is

```math
(A_i)_{rs}
=
\frac{\exp\!\left((Q_i)_r^{\mathsf T}(K_i)_s/\sqrt{d_h}\right)}
{\sum_{t=0}^{n_i}
\exp\!\left((Q_i)_r^{\mathsf T}(K_i)_t/\sqrt{d_h}\right)}.
```

The head output is

```math
O_i=A_iV_i.
```

Multihead attention concatenates head outputs and projects them back to width
d. Denote the resulting map by MHA.

```math
\mathrm{MHA}(Z_i)
=
\mathrm{Concat}\!\left(O_i^{(1)},\ldots,O_i^{(H)}\right)W_O.
```

### 3.2 The class-token row

The class row of the attention output is

```math
(O_i)_0
=
\sum_{s=0}^{n_i}
(A_i)_{0s}(V_i)_s.
```

Separating the self term from patch terms gives

```math
(O_i)_0
=
(A_i)_{00}(V_i)_0
+
\sum_{j=1}^{n_i}
(A_i)_{0j}(V_i)_j.
```

The coefficients in the second term are nonnegative and sum to at most one.
They sum to one only after including the class self coefficient. Thus the
class row is not necessarily a convex combination of patch values: it can
retain its previous latent state through the self term, and the output
projection can mix heads with signed coefficients.

If a residual connection is used, the pre-normalization class state has the
form

```math
\bar q_i
=
q_i^{\mathrm{in}}
+
\left[\mathrm{MHA}(Z_i)\right]_0.
```

An MLP then gives

```math
q_i^{\mathrm{out}}
=
\bar q_i
+
W_2\,\sigma\!\left(W_1\,\mathrm{LN}(\bar q_i)+b_1\right)+b_2.
```

The residual and MLP are part of the readout map. Ignoring them and calling
the class row a weighted average is only a first-order description.

## 4. When CLS reduces to attention pooling

### 4.1 Restricted construction

Consider a one-layer, single-head model with these restrictions:

1. the class query is fixed and independent of the patch bag;
2. patch values are computed pointwise from each patch;
3. patch-to-patch contextualization is disabled;
4. the class self term and output projection are removed or absorbed into a
   fixed affine map;
5. no positional or spatial operator changes the patch values.

Then the class output is

```math
z_i
=
\sum_{j=1}^{n_i}
\alpha_{ij}\,v(h_{ij}),
\qquad
\alpha_{ij}
=
\frac{\exp\!\left(q^{\mathsf T}k(h_{ij})/\sqrt{d_h}\right)}
{\sum_{\ell=1}^{n_i}
\exp\!\left(q^{\mathsf T}k(h_{i\ell})/\sqrt{d_h}\right)}.
```

This is attention pooling with a learned query. It is permutation invariant
because the same scalar score function is applied to every instance and the
sum is commutative.

The equivalence is useful, but it identifies a subfamily of CLS architectures,
not the full transformer.

### 4.2 Relation to gated attention MIL

Gated attention MIL commonly uses a score

```math
s(h_j)
=
w^{\mathsf T}
\left[
\tanh(Vh_j)
\odot
\mathrm{sigm}(Uh_j)
\right],
\qquad
\alpha_j
=
\frac{\exp(s(h_j))}
{\sum_{\ell=1}^{n_i}\exp(s(h_\ell))}.
```

and readout

```math
z_i^{\mathrm{ABMIL}}
=
\sum_{j=1}^{n_i}\alpha_j h_j.
```

Both formulas create normalized instance weights. They differ in at least
four ways:

- the gated MIL score is an explicit pointwise function of one patch;
- a CLS key score can use a learned query and a different value projection;
- a standard transformer can update patch values through patch-patch attention;
- the class state has residual memory across layers.

Therefore equal-looking softmax coefficients do not imply equal inductive
biases or equal explanations.

### 4.3 Relation to Set Transformer pooling by multihead attention

Set Transformer introduces learned seed vectors for pooling by multihead
attention. With one seed and no positional signal, a CLS token is structurally
close to one seed vector followed by attention to the set. The distinction in
a TransMIL-like architecture is that the token sequence also contains patch
self-attention and spatial PPEG operations.

The set-pooling subcase can be written as

```math
z_i
=
\mathrm{MAB}\!\left(q^{(0)},H_i\right).
```

where the seed query is fixed and the set rows are not assigned coordinate
identities. A TransMIL block instead acts on the augmented sequence:

```math
Z_i^{(\ell+1)}
=
\mathcal B_\ell\!\left(
\mathcal C_\ell\!\left(
Z_i^{(\ell)}
\right)
\right),
```

where C can include PPEG and B includes self-attention, residuals,
normalization, and an MLP. The class row is one coordinate of this joint state,
not an isolated pooling call.

## 5. Patch-patch context changes the readout

Let the first attention block produce contextual patch values

```math
\widetilde v_{ij}
=
\sum_{k=0}^{n_i}
(A_i)_{jk}(V_i)_k,
\qquad
1\le j\le n_i.
```

The class row in the next block can then be expressed schematically as

```math
q_i^{(2)}
=
\sum_{j=1}^{n_i}
\alpha_{ij}^{(2)}
\widetilde v_{ij}
+
\alpha_{i0}^{(2)}\widetilde v_{i0}
+\text{residual and MLP terms}.
```

Substitution exposes two-hop paths:

```math
q_i^{(2)}
\supseteq
\sum_{j=1}^{n_i}
\alpha_{ij}^{(2)}
\sum_{k=1}^{n_i}
\alpha_{ijk}^{(1)}
v(h_{ik}).
```

The effective coefficient of patch k is therefore a sum over intermediate
patches:

```math
\gamma_{ik}
=
\sum_{j=1}^{n_i}
\alpha_{ij}^{(2)}\alpha_{ijk}^{(1)}
\quad
\text{plus paths through residuals, PPEG, and the class row}.
```

Even when all direct class-to-patch coefficients are small, a patch can matter
through patch-patch routes. The final attention row only records one layer's
direct routing, not the full gamma coefficient.

For L layers, the number of computational paths grows combinatorially. Writing
the network as a directed acyclic computation graph, let P(k to 0) be paths
from patch k to the final class row. A local linearization gives

```math
\frac{\partial q_{i,0}^{(L)}}{\partial h_{ik}}
\approx
\sum_{\pi\in\mathcal P(k\rightsquigarrow 0)}
\prod_{e\in\pi}J_e,
```

where J_e is the Jacobian of an edge in the computation graph. Attention
weights appear in some J_e, but they are not the whole product.

## 6. Permutation invariance and its loss

### 6.1 No positional operator

Let P_i permute the patch rows while leaving the class row fixed:

```math
\widetilde P_i
=
\begin{bmatrix}
1&0\\
0&P_i
\end{bmatrix}.
```

For a self-attention block with shared rowwise projections and no positional
signal,

```math
\mathrm{SA}(\widetilde P_iZ_i)
=
\widetilde P_i\,\mathrm{SA}(Z_i).
```

The class row is fixed by the block permutation, so

```math
\left[\mathrm{SA}(\widetilde P_iZ_i)\right]_0
=
\left[\mathrm{SA}(Z_i)\right]_0.
```

By induction over pointwise MLPs, residuals, and layer normalization, the final
class token is permutation invariant when no position-dependent operation is
present:

```math
q_i^{(L)}(P_iH_i)=q_i^{(L)}(H_i).
```

This is the formal reason a CLS readout can represent a set model.

### 6.2 PPEG-conditioned model

With a fixed raster assignment Gamma_i, the permutation acts on feature values
but not on grid locations. PPEG then yields, in general,

```math
\mathrm{PPEG}_{\Gamma_i}\!\left(\mathcal R(P_iH_i)\right)
\ne
P_i\,\mathrm{PPEG}_{\Gamma_i}\!\left(\mathcal R(H_i)\right).
```

The final class token need not satisfy set invariance:

```math
q_i^{(L)}(P_iH_i;\Gamma_i)
\ne
q_i^{(L)}(H_i;\Gamma_i).
```

The distinction is not caused by the class token itself. It is caused by the
context operator coupled to the readout. A class token can be a permutation
invariant readout or an order-sensitive readout depending on what occurs before
the classifier.

## 7. The class token as a recurrent state

Define the class state and patch state at layer ell by

```math
q_i^{(\ell)}
=
Z_{i,0}^{(\ell)},
\qquad
H_i^{(\ell)}
=
Z_i^{(\ell)}[1:(n_i+1),:].
```

One transformer layer induces a coupled recurrence

```math
\begin{aligned}
q_i^{(\ell+1)}
&=
F_q^{(\ell)}
\left(q_i^{(\ell)},H_i^{(\ell)}\right),\\
H_i^{(\ell+1)}
&=
F_H^{(\ell)}
\left(q_i^{(\ell)},H_i^{(\ell)}\right).
\end{aligned}
```

The class row is therefore a memory state that is updated from the bag while
also sending information back to patch rows. It is not a passive accumulator.

The feedback path is visible in the attention matrix. Patch row j can attend
to the class row:

```math
\left[\mathrm{SA}(Z_i)\right]_j
\supseteq
(A_i)_{j0}(V_i)_0.
```

This makes the class token a global scratch space. Later patch updates can use
evidence already summarized by the class state, and later class updates can use
those recontextualized patch states.

In an architecture that masks patch-to-class feedback, the recurrence reduces
to a more recognizable pooling pattern. In ordinary full self-attention, it
does not.

## 8. A linearized information analysis

To study the bottleneck without nonlinear distractions, replace one transformer
block by a linear map on concatenated rows. Let

```math
\begin{bmatrix}
q_i^{(1)}\\
H_i^{(1)}
\end{bmatrix}
=
\begin{bmatrix}
M_{qq}&M_{qH}\\
M_{Hq}&M_{HH}
\end{bmatrix}
\begin{bmatrix}
q^{(0)}\\
H_i
\end{bmatrix}.
```

The class update is

```math
q_i^{(1)}
=
M_{qq}q^{(0)}
+
M_{qH}H_i.
```

Since q_i^(1) is in R^d while H_i is in R^(n_i times d), the linear map M_qH
has rank at most d. For n_i greater than one, its null space is generally
nontrivial:

```math
\exists\,\Delta H_i\ne 0
\quad\text{such that}\quad
M_{qH}\Delta H_i=0.
```

The two bags H_i and H_i plus Delta H_i then collide at the class state in
this linearized model. Nonlinear attention can enlarge the function class, but
the final map still returns only d numbers before the task head.

This is not automatically a defect. A supervised task only requires a statistic
that is sufficient for the target, not an injective reconstruction of the
entire bag. The relevant question is whether the lost directions contain
label-relevant information.

For a target Y_i, a representation Z_i is task-sufficient in the population if

```math
\mathbb P(Y_i\mid H_i)
=
\mathbb P(Y_i\mid Z_i).
```

The class token is a learned attempt to approximate such a statistic under
finite capacity and finite data. There is no architectural guarantee that it
does so.

## 9. Which statistics can collide?

### 9.1 Equal weighted means

Two bags can have the same attention-pooled mean:

```math
\sum_{j=1}^{n}\alpha_j h_j
=
\sum_{j=1}^{m}\beta_j h'_j,
\qquad
\sum_j\alpha_j=\sum_j\beta_j=1,
```

while having different counts, variances, or rare subpopulations. A one-vector
readout can preserve the relevant distinction only if the learned nonlinear
context maps it into the surviving coordinates.

### 9.2 Rare positive dilution

Suppose a positive signal appears in one patch and the remaining n minus one
patches are background. If the class attention coefficient on the positive
patch is small, its direct contribution is bounded by

```math
\left\|\alpha_{+}v(h_{+})\right\|_2
\le
\alpha_{+}\left\|v(h_{+})\right\|_2.
```

Patch-patch context can increase alpha-plus indirectly, but it can also spread
background evidence into the same representation and make the positive
direction harder to isolate.

### 9.3 Count and prevalence

If two bags contain the same patch types but different multiplicities, a
normalized attention readout can be insensitive to count in regimes where the
score distribution is unchanged. A class token can recover count through
normalization statistics, residual paths, or contextual interactions, but none
of these should be assumed without a count-controlled test.

## 10. Why the final attention row is not an explanation

Let the final prediction be

```math
\widehat y_i
=
\psi\!\left(q_i^{(L)}\right).
```

The final class-to-patch attention row is

```math
a_{ij}^{(L)}
=
\left(A_i^{(L)}\right)_{0j}.
```

This row answers a narrow question:

> At the final attention operation, how much normalized value mass did the
> class query route through each current token?

It does not answer:

- how much the patch changed the key and value vectors;
- how much earlier patch-patch attention changed that token;
- how the residual and MLP paths contributed;
- whether a high-attention patch was necessary for the prediction;
- whether deleting the patch changes the prediction;
- whether the patch is biologically causal or merely correlated.

The gradient of the prediction with respect to patch h_ij is

```math
g_{ij}
=
\frac{\partial \widehat y_i}{\partial h_{ij}}
=
\frac{\partial \psi}{\partial q_i^{(L)}}
\frac{\partial q_i^{(L)}}{\partial h_{ij}}.
```

The second factor includes all attention, residual, normalization, MLP, PPEG,
and class-feedback paths. In general,

```math
a_{ij}^{(L)}
\ne
\left\|g_{ij}\right\|,
\qquad
a_{ij}^{(L)}
\ne
\left|\widehat y_i-\widehat y_i^{(-j)}\right|.
```

Here y-hat minus j denotes a prediction after deleting or masking patch j.
Attention, gradient, and deletion are different observables.

## 11. Direct and indirect credit

For a one-layer class row without spatial context, the differential of the
attention output contains three terms for patch j:

```math
dq_0
=
\sum_{j=1}^{n}
\left[
(d\alpha_j)v_j
+
\alpha_j\,dv_j
\right]
+
(d\alpha_0)v_0
+
\alpha_0\,dv_0.
```

The first patch term changes the normalized routing coefficient; the second
changes the value content. Since the coefficient depends on the query and key,
perturbing a patch can also change every denominator term:

```math
d\alpha_\ell
=
\alpha_\ell
\left(
d\ell_\ell
-
\sum_{r=0}^{n}\alpha_r\,d\ell_r
\right).
```

Consequently, deleting one patch can redistribute attention over all other
patches. A local explanation that assigns only the removed patch's direct
coefficient ignores this competition effect.

For multiple layers, define the Jacobian blocks

```math
J_{ab}^{(\ell)}
=
\frac{\partial Z_{i,a}^{(\ell+1)}}
{\partial Z_{i,b}^{(\ell)}}.
```

The full patch-to-class Jacobian is

```math
\frac{\partial Z_{i,0}^{(L)}}{\partial Z_{i,j}^{(0)}}
=
\sum_{r_1,\ldots,r_{L-1}}
J_{0r_{L-1}}^{(L-1)}
\cdots
J_{r_1j}^{(0)}.
```

This path expansion is why attention rollout, integrated gradients, and
deletion tests can disagree. They aggregate different functions of the
Jacobian graph.

## 12. Class-token readout in TransMIL

The released TransMIL implementation uses the following relevant dimensions:

- input patch feature width: 1024;
- transformer width: 512;
- eight attention heads;
- two TransLayer modules;
- a learned class token of width 512;
- PPEG applied between the two transformer layers;
- LayerNorm on the final class row before the classifier.

Writing T_1 and T_2 for the two TransLayer maps, the paper-specific forward
path is

```math
Z_i^{(0)}
=
\begin{bmatrix}
q^{(0)}\\
\mathrm{ReLU}\!\left(X_iW_{\mathrm{in}}^{\mathsf T}+b_{\mathrm{in}}\right)
\end{bmatrix}.
```

The first layer and the intermediate spatial map are

```math
Z_i^{(1)}
=
\mathcal T_1\!\left(Z_i^{(0)}\right),
\qquad
\widetilde Z_i^{(1)}
=
\mathrm{PPEG}\!\left(Z_i^{(1)}\right).
```

The PPEG expression acts on patch rows and leaves the class row outside the
grid. The second layer gives

```math
Z_i^{(2)}
=
\mathcal T_2\!\left(\widetilde Z_i^{(1)}\right).
```

The final prediction is

```math
z_i
=
\mathrm{LN}\!\left(Z_{i,0}^{(2)}\right),
\qquad
\widehat y_i
=
W_{\mathrm{cls}}z_i+b_{\mathrm{cls}}.
```

This makes the readout order explicit:

```math
\text{patch features}
\xrightarrow{\text{attention}}
\text{contextual patch and class states}
\xrightarrow{\text{PPEG}}
\text{geometry-conditioned states}
\xrightarrow{\text{attention}}
\text{final class state}
\xrightarrow{\mathrm{LN}+\text{linear head}}
\text{slide prediction}.
```

The classifier never sees a separately normalized mean, max, or top-k list. It
sees one learned latent state after two rounds of coupled context formation.

## 13. Readout capacity and number of class tokens

One class token returns one vector. A multi-token readout returns r vectors:

```math
Q^{(0)}
=
\begin{bmatrix}
q_1^{(0)\mathsf T}\\
\vdots\\
q_r^{(0)\mathsf T}
\end{bmatrix}
\in\mathbb R^{r\times d}.
```

If these tokens attend to the same patch set, the output is

```math
Z_i^{\mathrm{read}}
=
\mathrm{MHA}\!\left(Q^{(0)},H_i,H_i\right)
\in\mathbb R^{r\times d}.
```

The scalar r controls how many learned summaries can be retained before a task
head. One token is efficient and easy to classify, but it can force distinct
tissue programs into the same state. Multiple tokens can represent different
modes, stains, regions, or task factors, but they increase capacity and make
the readout less automatically aligned with one scalar explanation.

A downstream head can pool the r token states:

```math
z_i
=
\mathrm{Pool}\!\left(
Z_i^{\mathrm{read}}
\right).
```

The choice of one versus many tokens is therefore a surviving-statistic choice,
not merely an implementation detail.

## 14. Failure modes

### 14.1 CLS starvation

If attention logits for the class query are nearly uniform, the class update
approaches a mean of values. Rare discriminative patches are diluted.

```math
\max_j\alpha_{ij}
\approx
\frac{1}{n_i}
\quad\Longrightarrow\quad
z_i
\text{ is close to a diffuse average}.
```

This can happen when key-query scale is too small, features are poorly
separated, or regularization suppresses score variance.

### 14.2 Attention saturation

If one logit dominates,

```math
\alpha_{ij^\star}\to 1,
\qquad
\alpha_{ij}\to 0
\quad
j\ne j^\star,
```

the model behaves like hard selection. This may be appropriate for a genuinely
single-instance task, but it is brittle under artifacts, patch duplication,
and small localization errors.

### 14.3 Context contamination

A high-scoring patch can attend to background or artifact tokens before the
class token reads it. The final class state may then represent a contextually
altered value rather than the original patch evidence.

### 14.4 Class-token shortcut

The class state can encode bag size, tissue fraction, scanner identity, or
padding convention through global statistics unrelated to the target. Because
the classifier sees only the final class state, this shortcut can be hard to
detect from heatmaps.

### 14.5 Single-vector collision

Different tissue compositions can produce nearly identical final class states.
The collision is especially plausible when the task requires several
independent rare signals but the representation width and number of readout
tokens are small.

### 14.6 Layerwise explanation mismatch

The final class row may look plausible while earlier layers route evidence
differently. Reporting it as the explanation of the whole model overstates what
the observable establishes.

### 14.7 Padding duplication

Repeated completion tokens can be counted multiple times by both attention and
PPEG. A class token that is sensitive to the completion remainder has learned a
numerical completion convention in addition to tissue evidence.

## 15. Toy counterexamples

### 15.1 Same final mean, different rare-event structure

Let a scalar classifier use a uniform class readout. The bags

```math
\mathcal B_1=\{1,1,1,1\},
\qquad
\mathcal B_2=\{4,0,0,0\}
```

have different maxima and different positive-instance interpretations, but
their means are both one. A one-dimensional class state initialized to the mean
cannot distinguish them without another contextual statistic.

### 15.2 Same direct attention, different context

Suppose two slides produce the same final class-to-patch coefficients, but one
slide has different patch-patch attention in the previous layer. Its contextual
values differ:

```math
\alpha_{ij}^{(L)}=\alpha_{ij}^{\prime(L)}
\quad\text{for all }j,
\qquad
\widetilde v_{ij}\ne\widetilde v_{ij}'.
```

The final predictions can still differ. Equal final attention maps do not imply
equal model decisions.

### 15.3 Duplicate sensitivity

Let a bag contain one positive token p and n minus one background tokens b.
Repeat p during completion. Even if the feature values are unchanged, its total
direct value mass can increase:

```math
\alpha_p v(p)
\quad\longmapsto\quad
\sum_{r\in\mathcal D_p}\alpha_r v(p).
```

The model can therefore change its prediction because of multiplicity created
by padding, not because new tissue evidence appeared.

## 16. Sanity checks for a CLS readout

### Check A: permutation test

For a model without position, randomly permute patch rows at inference time.
Predictions should remain numerically stable up to floating-point noise:

```math
\widehat y_i(P_iH_i)
\approx
\widehat y_i(H_i).
```

For a PPEG-equipped model, a change is expected; the question is whether it
tracks a physically valid raster permutation.

### Check B: direct-row versus deletion

Rank patches by final class attention and by leave-one-out prediction change.
Report rank correlation rather than assuming equality.

```math
\rho_{\mathrm{attn,del}}
=
\mathrm{corr}\!\left(
\{a_{ij}^{(L)}\}_j,
\{\widehat y_i-\widehat y_i^{(-j)}\}_j
\right).
```

Low correlation is not a bug by itself. It is evidence that the final row is
not a sufficient explanation observable.

### Check C: gradient agreement

Compare attention with gradient-times-input or integrated-gradient scores. A
large disagreement identifies strong value, key, residual, or MLP effects.

### Check D: class-token ablation

Replace the learned class token by a fixed zero vector, a random vector, or a
mean-pool initialization. If performance remains unchanged, the learned query
may not be carrying the intended readout role.

### Check E: readout-width sweep

Compare one, two, and several readout tokens while holding the patch encoder and
context budget fixed. The curve tests whether the one-vector bottleneck is
limiting the task.

### Check F: bag-composition intervention

Construct synthetic bags with controlled prevalence, count, and rare positives.
Measure the prediction as each statistic changes separately:

```math
\widehat y_i
=
f\!\left(
\text{positive prevalence},
\text{bag count},
\text{background distribution},
\text{spatial arrangement}
\right).
```

The intervention makes shortcuts visible that a slide-level test set can hide.

## 17. C/R/G/S placement

| Component | CLS-token transformer realization |
|---|---|
| Context operator | Self-attention over the augmented patch-plus-class sequence, optionally coupled to PPEG or another positional operator. |
| Readout operator | One learned class query updated through attention, residuals, normalization, and an MLP; the final class row is sent to the task head. |
| Geometry | Set-invariant when all context operators are permutation equivariant; order- or grid-conditioned when PPEG or another positional map is present. |
| Surviving statistic | A width-d latent state, or r such states for a multi-token readout; it preserves only task-relevant information that training manages to encode. |

The compact forward description is

```math
\boxed{
\text{patch set or raster}
\xrightarrow{\text{joint context}}
\text{coupled patch/class states}
\xrightarrow{\text{class-state selection}}
\text{slide statistic}
\xrightarrow{\text{task head}}
\text{prediction}
}
```

The readout operator is therefore not separable from the context operator. The
same class token can realize set pooling, spatially conditioned pooling, or a
recurrent global memory depending on what occurs before the classifier.

## 18. Bottom line

A CLS token is best understood as a learned query and a recurrent latent state.
It can collapse to attention pooling in a restricted architecture, but a
TransMIL-style network goes beyond that reduction because patch-patch attention,
PPEG, residual paths, normalization, and MLPs all shape the final class vector.

The mathematically honest statement is:

```math
\text{CLS readout}
\ne
\text{final attention row alone}
\ne
\text{causal patch explanation}.
```

The class token is a useful compression interface between a large patch
sequence and a slide-level head. Its success depends on whether that compressed
state preserves the target-relevant composition, context, and geometry of the
slide. Its explanation requires interventions or full-path attribution, not
just a heatmap of the last routing coefficients.

## References

- Shao et al., "TransMIL: Transformer based Correlated Multiple Instance
  Learning for Whole Slide Image Classification," NeurIPS 2021.
  https://arxiv.org/abs/2106.00908
- Lee et al., "Set Transformer: A Framework for Attention-based Permutation-
  Invariant Neural Networks," ICML 2019.
  https://arxiv.org/abs/1810.00825
- Dosovitskiy et al., "An Image is Worth 16x16 Words: Transformers for Image
  Recognition at Scale," ICLR 2021.
  https://arxiv.org/abs/2010.11929
