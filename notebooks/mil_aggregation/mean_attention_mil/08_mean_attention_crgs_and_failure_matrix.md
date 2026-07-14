# Mean and Attention MIL C/R/G/S Design Matrix

This note is the synthesis layer for the mean and attention MIL family. It
does not replace the derivations of mean pooling, ABMIL, CLAM, DSMIL, additive
MIL, or DTFD-MIL. Its purpose is to place them on the same mathematical axes:

```math
\text{context}
\longrightarrow
\text{readout}
\longrightarrow
\text{surviving statistic}
\longrightarrow
\text{failure mode}.
```

The central design question is not which module has the most recognizable name.
It is which information the composition preserves and which equivalence classes
of slides it collapses.

## 1. Common bag notation

Let slide i contain n_i patch features:

```math
H_i
=
\{h_{ij}\}_{j=1}^{n_i},
\qquad
h_{ij}\in\mathbb R^d.
```

A context operator transforms each instance:

```math
u_{ij}
=
\mathcal C_j(H_i)
\in
\mathbb R^p.
```

A readout maps the transformed bag to a slide representation:

```math
z_i
=
\mathcal R
\left(
\{u_{ij}\}_{j=1}^{n_i}
\right).
```

A task head produces the prediction:

```math
\widehat y_i
=
\mathcal H(z_i).
```

The C/R/G/S decomposition is:

```math
H_i
\xrightarrow{\mathcal C}
U_i
\xrightarrow{\mathcal R}
z_i
\xrightarrow{\mathcal H}
\widehat y_i,
```

with geometry G and supervision S specifying how C, R, and H are constructed
and trained.

The same readout can mean different things under different context operators.
A weighted mean over independent patches is not the same statistic as a
weighted mean over graph-updated or transformer-updated patch states.

## 2. Context operator C

### 2.1 None or shared instance map

The simplest context operator is shared feature transformation:

```math
u_{ij}
=
\phi(h_{ij}).
```

It is pointwise:

```math
u_{ij}
=
u_{ij}(h_{ij}),
```

so the representation of patch j does not depend on other patches except
through a later normalization in the readout.

Mean pooling and classic ABMIL are in this class.

### 2.2 Attention score context

ABMIL adds a score computed from each transformed instance:

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

The score is pointwise, but the normalized coefficient is contextual through
the bag denominator:

```math
a_{ij}
=
\frac{\exp(s_{ij})}
{\sum_{\ell=1}^{n_i}\exp(s_{i\ell})}.
```

Thus the score network is local while the normalized readout is competitive.

### 2.3 Class-specific attention context

CLAM creates a branch for class c:

```math
s_{ij}^{(c)}
=
w_c^{\mathsf T}
\left[
\tanh(V_cu_{ij})
\odot
\sigma(U_cu_{ij})
\right].
```

The branch coefficient is:

```math
a_{ij}^{(c)}
=
\frac{\exp(s_{ij}^{(c)})}
{\sum_{\ell=1}^{n_i}\exp(s_{i\ell}^{(c)})}.
```

The context remains pointwise before normalization, but the target index c is
now part of the routing mechanism.

### 2.4 Critical-instance context

DSMIL first chooses a critical instance:

```math
j_i^\star
=
\arg\max_j
\rho_{ij},
```

then uses it to condition the remaining instance scores:

```math
s_{ij}
=
\mathrm{sim}
\left(
q(h_{ij}),
k(h_{i,j_i^\star})
\right).
```

Now the transformed score for one patch depends on another patch. The context
operator is relational even if the final aggregation is attention-weighted.

### 2.5 Pseudo-bag context

DTFD partitions a slide:

```math
\{1,\ldots,n_i\}
=
\biguplus_{m=1}^{M}\mathcal B_{i,m}.
```

Tier 1 acts within each pseudo-bag:

```math
F_{i,m}^{(1)}
=
\mathcal R_1
\left(
\{h_{ij}:j\in\mathcal B_{i,m}\}
\right).
```

Tier 2 acts on distilled pseudo-bag features:

```math
F_i^{(2)}
=
\mathcal R_2
\left(
\{\widehat f_{i,m}\}_{m=1}^{M}
\right).
```

This is a compositional context operator. Vanilla random pseudo-bag membership
is not physical spatial context.

## 3. Readout operator R

### 3.1 Mean

The normalized first moment is:

```math
z_i^{\mathrm{mean}}
=
\frac{1}{n_i}
\sum_{j=1}^{n_i}u_{ij}.
```

It is permutation invariant:

```math
z_i^{\mathrm{mean}}
\left(
u_{i,\pi(1)},\ldots,u_{i,\pi(n_i)}
\right)
=
z_i^{\mathrm{mean}}
\left(
u_{i,1},\ldots,u_{i,n_i}
\right).
```

It is also replication invariant when every instance is duplicated equally:

```math
z^{\mathrm{mean}}(H_i\uplus H_i)
=
z^{\mathrm{mean}}(H_i).
```

That invariance is a modeling assumption. If tissue exposure or lesion count
matters, normalized mean pooling can erase it.

### 3.2 Sum

The additive first moment is:

```math
z_i^{\mathrm{sum}}
=
\sum_{j=1}^{n_i}u_{ij}.
```

Under equal replication:

```math
z^{\mathrm{sum}}(H_i\uplus H_i)
=
2z^{\mathrm{sum}}(H_i).
```

Sum pooling retains count and exposure but is sensitive to patch count and
slide preparation. Mean and sum are not interchangeable normalization choices.

### 3.3 Max

For scalar evidence r_ij:

```math
z_i^{\mathrm{max}}
=
\max_j r_{ij}.
```

Max retains a witness-like upper-tail statistic. It discards all submaximal
values except insofar as they determine the runner-up in a deletion test.

### 3.4 Attention-weighted mean

ABMIL uses:

```math
z_i^{\mathrm{attn}}
=
\sum_{j=1}^{n_i}a_{ij}v_{ij},
\qquad
a_{ij}
=
\frac{\exp(s_{ij})}
{\sum_{\ell=1}^{n_i}\exp(s_{i\ell})}.
```

This is a learned weighted first moment. The weights are nonnegative and sum to
one, so the representation is in the convex hull of the value vectors.

### 3.5 Class-specific attention

CLAM has one weighted first moment per class branch:

```math
z_i^{(c)}
=
\sum_{j=1}^{n_i}
a_{ij}^{(c)}v_{ij}^{(c)}.
```

Class-specific routing can preserve different evidence for different outputs,
but it also means attention maps are not target-free objects.

### 3.6 Critical plus context

A DSMIL-like readout can combine a critical branch and a context branch:

```math
z_i^{\mathrm{DSMIL}}
=
\lambda_i z_i^{\mathrm{critical}}
+
(1-\lambda_i)z_i^{\mathrm{context}}.
```

The surviving statistic is neither a pure maximum nor a pure mean. It is an
extreme-conditioned contextual summary.

### 3.7 Additive signed evidence

An additive MIL logit is:

```math
\ell_i^{\mathrm{add}}
=
b+\sum_{j=1}^{n_i}r_{ij}.
```

Unlike attention coefficients, r_ij can be negative. The readout retains
signed total evidence, not a convex average.

### 3.8 DTFD distillation

For each pseudo-bag, DTFD applies one of four operators:

```math
\widehat f_{i,m}^{\mathrm{MaxS}}
=
h_{i,\arg\max_{j\in\mathcal B_{i,m}}p_{i,m,j}^{1}},
```

```math
\widehat f_{i,m}^{\mathrm{MaxMinS}}
=
\left[
h_{i,j_{i,m}^{+}}
\middle\Vert
h_{i,j_{i,m}^{-}}
\right],
```

```math
\widehat f_{i,m}^{\mathrm{MAS}}
=
h_{i,\arg\max_{j\in\mathcal B_{i,m}}a_{i,m,j}},
```

```math
\widehat f_{i,m}^{\mathrm{AFS}}
=
\sum_{j\in\mathcal B_{i,m}}
a_{i,m,j}h_{ij}.
```

Tier 2 then applies attention or another readout to the distilled collection.
The method retains a tiered evidence statistic, not the original patch
distribution.

## 4. Geometry operator G

### 4.1 No geometry

A flat set model receives:

```math
G_i
=
\varnothing.
```

Any permutation of the feature collection is equivalent to the model.

### 4.2 Coordinates for rendering only

A model may retain coordinates c_ij outside the predictor:

```math
\mathrm{render}(c_{ij},\mathrm{score}_{ij}).
```

This gives spatial visualization, not spatial reasoning. The prediction remains
unchanged under coordinate permutation if coordinates are not fed to C.

### 4.3 Coordinates as features

Coordinates can enter the instance state:

```math
u_{ij}
=
\phi
\left(
[h_{ij}\Vert c_{ij}]
\right).
```

This makes location available to the model, but it introduces positional
shortcuts and requires coordinate normalization across slides.

### 4.4 Random pseudo-bag membership

DTFD's vanilla allocation is:

```math
\mathcal B_{i,m}
=
\pi_i^{-1}
\left(
\text{index block }m
\right),
```

where pi-i is a random permutation. It creates computational groups but not
neighborhoods.

### 4.5 Relational critical context

DSMIL uses a feature relation to a selected instance:

```math
G_i^{\mathrm{critical}}
=
\left\{
(h_{ij},h_{i,j_i^\star})
\right\}_{j=1}^{n_i}.
```

This is feature geometry rather than physical geometry.

## 5. Supervision operator S

### 5.1 Slide-level label

The observed target is:

```math
Y_i
\in
\{0,1\}.
```

Under positive-instance MIL:

```math
Y_i
=
\mathbf 1
\left\{
\sum_jZ_{ij}>0
\right\}.
```

The latent Z values are not observed.

### 5.2 Mean and ABMIL supervision

The slide loss is applied after readout:

```math
\mathcal L
=
\sum_i
\mathrm{CE}
\left(
\mathcal H(z_i),Y_i
\right).
```

No local target is explicitly constructed.

### 5.3 CLAM pseudo-label supervision

CLAM selects high- and low-attention instances:

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
\right).
```

These are generated pseudo-label sets, not pathologist annotations. Their
errors can feed back into the attention branch.

### 5.4 DSMIL critical supervision

The critical index is generated from a class score:

```math
j_i^\star
=
\arg\max_j\rho_{ij}.
```

The index is a learned routing decision, not a known positive-instance label.

### 5.5 Additive supervision

Additive evidence is trained only through the slide objective unless an
additional local loss is supplied:

```math
\mathcal L_{\mathrm{add}}
=
\sum_i
\mathrm{CE}
\left(
\sigma(\ell_i^{\mathrm{add}}),Y_i
\right).
```

A signed instance term does not become a ground-truth local label.

### 5.6 DTFD inherited pseudo-bag labels

DTFD uses:

```math
\widetilde Y_{i,m}
=
Y_i.
```

The latent pseudo-bag truth is:

```math
Y_{i,m}^{\star}
=
\mathbf 1
\left\{
\sum_{j\in\mathcal B_{i,m}}Z_{ij}>0
\right\}.
```

The mismatch event is:

```math
\widetilde Y_{i,m}=1,
\qquad
Y_{i,m}^{\star}=0.
```

This is generated weak supervision, not local annotation.

## 6. Surviving statistics

A method's surviving statistic is the information that can still affect the
prediction after the readout.

### Mean

```math
z_i^{\mathrm{mean}}
=
\mathbb E_{\widehat\mu_i}[U],
```

where the empirical measure is:

```math
\widehat\mu_i
=
\frac{1}{n_i}
\sum_{j=1}^{n_i}\delta_{u_{ij}}.
```

Mean pooling retains only the first moment.

### Sum

```math
z_i^{\mathrm{sum}}
=
n_i\mathbb E_{\widehat\mu_i}[U].
```

It retains the first moment multiplied by count.

### Max

```math
z_i^{\mathrm{max}}
=
\sup_j r_{ij}.
```

It retains an upper-tail witness.

### ABMIL

```math
z_i^{\mathrm{attn}}
=
\sum_j a_{ij}v_{ij}.
```

It retains a learned weighted first moment. The attention scores themselves
are not necessarily retained by the final head unless exposed separately.

### CLAM

```math
z_i^{\mathrm{CLAM}}
=
\left\{
z_i^{(c)}
\right\}_{c=1}^{C}.
```

It retains class-conditional weighted first moments plus any branch-specific
instance losses.

### DSMIL

```math
z_i^{\mathrm{DSMIL}}
=
\left(
h_{i,j_i^\star},
z_i^{\mathrm{context}}
\right).
```

It retains an extreme or critical feature and a context summary conditioned on
that feature.

### Additive MIL

```math
\ell_i^{\mathrm{add}}
=
b+\sum_jr_{ij}.
```

It retains a signed sum in score space.

### DTFD

```math
\widehat D_i
=
\{\widehat f_{i,m}\}_{m=1}^{M}.
```

It retains one selected or aggregated feature per pseudo-bag, followed by a
second-tier summary.

## 7. Collision classes

A collision occurs when two distinct bags receive the same representation:

```math
H_i\ne H_{i'}
\quad\text{but}\quad
\mathcal R(\mathcal C(H_i))
=
\mathcal R(\mathcal C(H_{i'})).
```

Once a collision occurs, no downstream head can distinguish the bags from that
representation alone.

### Mean collision

If two bags have the same empirical mean:

```math
\frac{1}{n_i}\sum_j u_{ij}
=
\frac{1}{n_{i'}}\sum_j u_{i'j},
```

mean pooling collides them even if their variances, multimodality, or tails
differ.

### Attention collision

Two bags can collide under ABMIL whenever their attention-weighted first
moments coincide:

```math
\sum_j a_{ij}v_{ij}
=
\sum_j a_{i'j}v_{i'j}.
```

The internal scores need not be identical.

### Additive collision

Two bags collide under an additive scalar head if:

```math
\sum_jr_{ij}
=
\sum_jr_{i'j}.
```

This can happen for different positive and negative evidence configurations.

### DTFD collision

If each pseudo-bag produces the same distilled feature collection:

```math
\{\widehat f_{i,m}\}_{m=1}^{M}
=
\{\widehat f_{i',m}\}_{m=1}^{M},
```

Tier 2 cannot distinguish the slides.

## 8. Operator contracts

A useful comparison states the invariance and sensitivity contract of each
operator.

| Operator | Permutation | Equal duplication | Count sensitivity | Tail sensitivity |
| --- | --- | --- | --- | --- |
| mean | invariant | invariant | low | low |
| sum | invariant | scales | high | low |
| max | invariant | invariant | low | high |
| ABMIL | invariant | depends on score/value behavior | indirect | learned |
| CLAM | invariant within branch | branch-dependent | indirect | class-dependent |
| DSMIL | invariant as a set | critical-switch dependent | indirect | high |
| additive | invariant | scales under duplication | high | depends on score |
| DTFD | partition-dependent realization | split-dependent | through M and m | rule-dependent |

The contract is more informative than the module name. “Attention MIL” does
not uniquely specify duplication behavior, target dependence, or whether context
is present before the readout.

## 9. Composition is order-sensitive

Let C be a context operator and R a readout. In general:

```math
\mathcal R\circ\mathcal C
\ne
\mathcal C\circ\mathcal R.
```

The right side may not even be type-correct. A graph update before attention
creates relational features, while attention before a graph update creates a
different graph input.

For two operators C-one and C-two:

```math
\mathcal R
\left(
\mathcal C_2
\left(
\mathcal C_1(H_i)
\right)
\right)
\ne
\mathcal R
\left(
\mathcal C_1
\left(
\mathcal C_2(H_i)
\right)
\right)
\quad
\text{in general}.
```

For DTFD, local readout and global readout are nested:

```math
\mathcal R_2
\left(
\left\{
\Phi_m
\left(
\mathcal R_1(H_{i,m})
\right)
\right\}_{m=1}^{M}
\right).
```

Replacing this by one flat readout changes the surviving statistic.

## 10. Mean versus attention

Mean pooling is the special case of attention with uniform coefficients:

```math
a_{ij}
=
\frac{1}{n_i}
\quad
\forall j.
```

Then:

```math
z_i^{\mathrm{attn}}
=
\sum_j
\frac{1}{n_i}v_{ij}
=
z_i^{\mathrm{mean}}.
```

The converse is not true. A learned attention model can represent uniform
attention, but it also represents nonuniform weighted first moments.

The distinction is not merely capacity. The learned coefficient creates
competition:

```math
\frac{\partial a_{ij}}{\partial s_{i\ell}}
=
a_{ij}
\left(
\mathbf 1\{j=\ell\}-a_{i\ell}
\right).
```

Mean pooling has no such learned score competition.

## 11. Attention concentration

Define entropy:

```math
\mathcal H_i
=
-
\sum_j a_{ij}\log a_{ij}.
```

Define effective support:

```math
N_{\mathrm{eff},i}
=
\frac{1}{\sum_j a_{ij}^{2}}.
```

A concentrated attention distribution has small effective support. This can be
useful for sparse witness labels, but concentration is not evidence of
faithfulness.

A diffuse distribution can preserve contextual burden, but it can dilute rare
positive instances. The label assumption determines whether concentration or
coverage is desirable.

## 12. Rare-positive dilution

Suppose one positive feature h-plus is mixed with negative weighted mean h-bar-minus.
The ABMIL representation is:

```math
z_i
=
a_+h_+
+
(1-a_+)\overline h_-.
```

For a target direction beta:

```math
\beta^{\mathsf T}z_i
=
a_+\beta^{\mathsf T}h_+
+
(1-a_+)\beta^{\mathsf T}\overline h_-.
```

If a-plus is small, a positive feature can be overwhelmed by the negative
context. This is the core dilution mechanism for a distributed first moment.

Max pooling avoids averaging this positive feature away if it receives the
largest score, but becomes sensitive to false extreme patches. Attention learns
where to place mass but can collapse or fail to discover the witness.

## 13. Distributed-burden bias

Suppose a slide label depends on the number or fraction of positive patches:

```math
Y_i
\approx
\mathbf 1
\left\{
\frac{1}{n_i}\sum_jZ_{ij}>\tau
\right\}.
```

A count-normalized mean is aligned with this assumption. A max operator is not:
one positive patch and many positive patches can have the same maximum.

For sum evidence:

```math
\ell_i^{\mathrm{sum}}
=
b+
\sum_jr_{ij},
```

the output can grow with burden and exposure. If n_i varies because of tissue
area rather than disease burden, this can become a confound.

The correct operator depends on whether count, fraction, or existence is the
latent label mechanism.

## 14. CLAM boundary

CLAM adds class-specific branches and an instance-level clustering objective.
Its design is represented as:

```math
H_i
\xrightarrow{\mathcal C=\mathrm{shared\ instance\ map}}
\{u_{ij}\}
\xrightarrow{\mathcal R_c=\mathrm{class\ attention}}
z_i^{(c)}
\xrightarrow{\mathcal H_c}
\widehat y_i^{(c)}.
```

The auxiliary selection is:

```math
\mathcal T_i^{(c)}
=
\mathrm{TopK}(a_{ij}^{(c)}),
\qquad
\mathcal B_i^{(c)}
=
\mathrm{BottomK}(a_{ij}^{(c)}).
```

Its surviving statistic is class-conditional attention plus selected instance
constraints. Its principal collision is two bags with the same class-branch
weighted means and the same auxiliary losses.

Its principal shortcut is confirmation bias: early attention errors can create
pseudo-labels that reinforce the same error.

## 15. DSMIL boundary

DSMIL adds a critical-instance relation:

```math
j_i^\star
=
\arg\max_j\rho_{ij},
```

```math
s_{ij}
=
\mathrm{sim}
\left(
q(h_{ij}),
k(h_{i,j_i^\star})
\right).
```

Its context operator is relational. Its readout combines a critical score branch
with an attention-weighted bag branch.

The key collision is two bags with the same critical feature and same
critical-conditioned context embedding, even if their noncritical patch
distributions differ.

The key failure is a critical-index switch:
a small score perturbation can change j-star and therefore all conditioned
scores.

## 16. Additive MIL boundary

Additive MIL decomposes the target logit:

```math
\ell_i
=
b+\sum_jr_{ij}.
```

Its signed statistic can represent both positive and negative evidence:

```math
r_{ij}>0
\quad
\text{and}
\quad
r_{i\ell}<0.
```

A softmax attention map cannot encode this sign without a separate value-space
or target-specific score. Additive MIL can also miss pairwise interactions:
two patches with individually weak scores may be jointly diagnostic.

An interaction extension is:

```math
\ell_i
=
b+\sum_jr_{ij}
+\sum_{j<\ell}r_{ij\ell}.
```

The added terms increase expressivity but remove the simple independent
contribution interpretation.

## 17. DTFD boundary

DTFD composes local and global readouts:

```math
H_i
\longrightarrow
\{H_{i,m}\}_{m=1}^{M}
\longrightarrow
\{F_{i,m}^{(1)}\}_{m=1}^{M}
\longrightarrow
\{\widehat f_{i,m}\}_{m=1}^{M}
\longrightarrow
F_i^{(2)}.
```

Its supervision has two levels:

```math
\mathcal L_1
=
\frac{1}{NM}
\sum_{i,m}
\mathrm{CE}(y_{i,m},Y_i),
```

```math
\mathcal L_2
=
\frac{1}{N}
\sum_i
\mathrm{CE}(\widehat y_i,Y_i).
```

The pseudo-bag target is noisy when a positive parent bag contains a
pseudo-bag with no positive patch. The probability for a pseudo-bag of size m
is:

```math
\Pr(Y_{i,m}^{\star}=0\mid Y_i=1)
=
\frac{
\binom{n_i-n_i^+}{m}
}{
\binom{n_i}{m}
}.
```

The main DTFD collision is identical distilled collections from different
slides. Its main geometric limitation is that random pseudo-bags do not
preserve adjacency.

## 18. Target-specific readout credit

For a target q, a local gradient credit is:

```math
G_{ij}^{(q)}
=
\left(
\frac{\partial q}{\partial u_{ij}}
\right)^{\mathsf T}
u_{ij}.
```

An attention coefficient is:

```math
a_{ij}.
```

A deletion effect is:

```math
\Delta_{ij}^{(q)}
=
q(H_i)-q(H_i\setminus j).
```

For softmax attention, the representation deletion identity is:

```math
z_i-z_{i,-j}
=
\frac{a_{ij}}{1-a_{ij}}
\left(
v_{ij}-z_{i,-j}
\right).
```

Therefore a high coefficient can have a small target effect if its value is
near the remaining mean, and a moderate coefficient can have a large effect if
its value is far from that mean in the target direction.

## 19. Failure matrix by label assumption

| Label assumption | Appropriate surviving statistic | Main collision | Main failure |
| --- | --- | --- | --- |
| one positive witness | max, high quantile, sparse attention | same selected witness | false extreme or missed rare patch |
| distributed fraction | mean or calibrated attention mean | same first moment | rare evidence dilution |
| total burden | sum or count-aware statistic | same signed total | exposure confounding |
| opposing evidence | signed additive score | same net score | cancellation hides mechanism |
| class-specific morphology | branch-specific weighted moments | same branch embeddings | pseudo-label confirmation |
| pairwise arrangement | relational context plus readout | same relational summary | missing interactions |
| spatial pattern | coordinate-aware C or G | same spatial embedding | flat model has spatial shortcut |
| pseudo-bag multiscale signal | tiered local/global readout | same distilled collection | random partition bias |

The table is a design aid. It does not claim that one operator solves one task
universally.

## 20. Toy collision: same mean, different slide

Take one-dimensional bags:

```math
H_A
=
\{-1,1\},
\qquad
H_B
=
\{0,0\}.
```

Their means are equal:

```math
z_A^{\mathrm{mean}}
=
z_B^{\mathrm{mean}}
=
0.
```

Their variances differ:

```math
\mathrm{Var}(H_A)=1,
\qquad
\mathrm{Var}(H_B)=0.
```

A mean-only head cannot distinguish a bimodal bag from a concentrated bag.
Moment pooling, attention, or a nonlinear set encoder is required if the
variance or multimodality is label-relevant.

## 21. Toy collision: same maximum, different burden

Let:

```math
H_A
=
\{1,0,0,0\},
\qquad
H_B
=
\{1,1,1,1\}.
```

Both have maximum one:

```math
\max(H_A)=\max(H_B)=1.
```

Their positive fractions differ. A max readout cannot represent distributed
burden from this pair.

## 22. Toy collision: same attention embedding, different tails

Let two bags have value vectors whose weighted means match:

```math
\sum_j a_{Aj}v_{Aj}
=
\sum_j a_{Bj}v_{Bj}.
```

Their attention entropy can still differ:

```math
\mathcal H_A
\ne
\mathcal H_B.
```

If the head sees only the weighted mean, it cannot distinguish concentrated
routing from diffuse routing. Exposing entropy or higher moments changes the
representation object.

## 23. Toy geometry failure

Let two slides have identical feature multisets and different coordinates:

```math
\{(h_j,c_j)\}_{j=1}^{n}
\ne
\{(h_j,c'_j)\}_{j=1}^{n},
```

with the same h values. If coordinates are not used by C or G:

```math
f(H,c)
=
f(H,c').
```

A coordinate-rendered heatmap can look spatial while the predictor is
invariant to the spatial arrangement. This is a visualization-model mismatch.

## 24. Noncommutativity of class selection and pooling

Let P be a class-specific projection and R be mean pooling. In general:

```math
R(P(H_i))
=
\frac{1}{n_i}\sum_jP(h_{ij}),
```

whereas selecting the top class score first and then pooling gives:

```math
R_{\mathrm{topK}}(H_i)
=
\frac{1}{K}
\sum_{j\in\mathrm{TopK}(H_i)}h_{ij}.
```

These operations do not commute:

```math
R(P(H_i))
\ne
R_{\mathrm{topK}}(H_i).
```

CLAM's top and bottom instance selection is therefore not equivalent to merely
adding a class projection before a flat mean.

## 25. Design axes not fixed by the module name

For each method, record:

1. whether C is pointwise or relational;
2. whether R is count-normalized;
3. whether R is class-conditioned;
4. whether selection is hard or soft;
5. whether G is physical, feature-based, random, or absent;
6. whether S is observed or generated;
7. which statistic survives;
8. whether the output is a representation, a logit, or a calibrated probability.

A compact formal record is:

```math
\mathfrak M
=
\left(
\mathcal C,
\mathcal R,
\mathcal G,
\mathcal S,
\mathcal H,
\mathcal I
\right),
```

where I is the intended invariance contract.

Two methods with the same attention equation can differ in G, S, or H and
therefore have different failure modes.

## 26. C/R/G/S matrix

| Method | C | R | G | S |
| --- | --- | --- | --- | --- |
| mean MIL | pointwise | normalized first moment | none | slide label |
| ABMIL | pointwise score plus competitive normalization | weighted first moment | none unless features include coordinates | slide label |
| CLAM | pointwise class branches | class-specific weighted moments and top/bottom selection | none by default | slide label plus generated cluster targets |
| DSMIL | critical-instance relational context | critical branch plus contextual attention | feature relation | slide label plus generated critical index |
| additive MIL | pointwise or optional context | signed score sum | inherited from features | slide label |
| DTFD-MIL | local pseudo-bag ABMIL then global ABMIL | selection or local weighted mean followed by Tier 2 | random partition in vanilla method | copied pseudo-bag label plus slide label |

The table should be read vertically. Changing only one column can change the
interpretation of the entire model.

## 27. Information loss audit

For a proposed readout, ask whether the following are preserved:

```math
\text{count},
\qquad
\text{mean},
\qquad
\text{variance},
\qquad
\text{tail},
\qquad
\text{class direction},
\qquad
\text{pairwise relation},
\qquad
\text{physical layout}.
```

Mean preserves one first moment. Max preserves one upper-tail value. Attention
preserves one learned first moment. CLAM preserves class-conditioned moments.
DSMIL preserves an extreme-conditioned context. Additive MIL preserves a signed
total. DTFD preserves a collection of distilled local summaries.

No readout preserves all seven by default.

## 28. How to test the claimed statistic

### 28.1 Count test

Duplicate every patch:

```math
H_i'
=
H_i\uplus H_i.
```

Compare:

```math
z(H_i')
\quad
\text{with}
\quad
z(H_i).
```

This diagnoses replication invariance.

### 28.2 Tail test

Insert one high-scoring patch and measure:

```math
z(H_i\uplus\{h_{\mathrm{extreme}}\})
-
z(H_i).
```

This tests witness sensitivity.

### 28.3 Distribution test

Construct bags with equal means and different variances. If the output is
unchanged, variance is not represented by the selected statistic.

### 28.4 Geometry test

Keep features fixed and permute coordinates. If the output changes, geometry
enters the model. If only the rendering changes, geometry is visualization-only.

### 28.5 Partition test

Resample the DTFD partition:

```math
\widehat D_i(\pi_i)
\quad
\text{versus}
\quad
\widehat D_i(\pi_i').
```

This measures random grouping sensitivity.

### 28.6 Target test

Compute attribution for two heads:

```math
\Delta_{ij}^{(q_1)}
\quad
\text{versus}
\quad
\Delta_{ij}^{(q_2)}.
```

A single target-free heatmap cannot stand in for both.

## 29. Complexity and sample dependence

Mean, sum, and flat attention process every instance. The attention readout has
normalization cost proportional to n_i:

```math
\mathcal O(n_i d)
```

up to score-network costs.

DTFD processes the same total number of patch features in Tier 1 when the
pseudo-bags partition the slide:

```math
\sum_{m=1}^{M}
|\mathcal B_{i,m}|
=
n_i.
```

It changes normalization granularity from n_i to approximately n_i over M,
then adds a second-tier cost over M:

```math
\mathcal O(n_i d)
+
\mathcal O(Md).
```

The computational gain can arise from smaller intermediate bags and training
organization, but the sample dependence remains at the parent-slide level.

## 30. Reporting rule

For every method, report the following sentence in mathematical form:

```math
\text{prediction}
=
\mathcal H
\left(
\mathcal R
\left(
\mathcal C
\left(
H_i;G_i,S_i
\right)
\right)
\right).
```

Then state:

```math
\text{surviving statistic}
=
\text{the statistic of }H_i\text{ retained by }\mathcal R\circ\mathcal C.
```

Then state the primary collision:

```math
\mathcal R(\mathcal C(H_i))
=
\mathcal R(\mathcal C(H_{i'}))
\Longrightarrow
\text{downstream head cannot distinguish }i\text{ and }i'.
```

This forces the discussion to name both the inductive bias and the information
it discards.

## 31. Bottom line

Mean and attention MIL are not a list of interchangeable pooling names. They
are compositions with different context, readout, geometry, and supervision
contracts:

```math
\boxed{
\text{model meaning}
=
(\mathcal C,\mathcal R,\mathcal G,\mathcal S)
+
\text{surviving statistic}
+
\text{collision class}
}
```

Mean pooling retains a normalized first moment. ABMIL learns a weighted first
moment. CLAM makes that moment class-specific and adds generated instance
constraints. DSMIL conditions context on a critical instance. Additive MIL
retains signed evidence in score space. DTFD creates a two-level summary whose
pseudo-bag labels are inherited and whose grouping is random in the vanilla
construction.

The design question is therefore:

```math
\text{What label-relevant information must survive aggregation?}
```

Only after answering that question should the attention, critical-instance,
class-branch, or distillation module be chosen.
