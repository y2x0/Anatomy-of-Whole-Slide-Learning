# Attention Support and Long-Range Interactions

## 1. Scope

This note asks a more precise version of the usual transformer claim:

> Does self-attention actually preserve long-range WSI interactions?

The answer depends on which object is being called support:

1. the mathematical support of the attention matrix;
2. the thresholded or entropy-defined effective support;
3. the support of the input-output Jacobian;
4. the support that survives an approximation such as Nyström attention;
5. the support that is useful for the task.

These are not interchangeable. A dense softmax matrix has positive entries
almost everywhere, but many entries can be numerically negligible. A sparse
effective support can still produce a globally connected computation graph
through a few bridge tokens. Conversely, a mathematically dense matrix can have
weak collective long-range signal that is destroyed by approximation or
normalization.

The paper-specific anchor is Shao et al., TransMIL: Transformer based
Correlated Multiple Instance Learning for Whole Slide Image Classification:

https://arxiv.org/abs/2106.00908

The general sparse-attention comparison uses Martins and Astudillo, Sparsemax:

https://arxiv.org/abs/1602.02068

and Peters et al., Sparse Sequence-to-Sequence Models:

https://arxiv.org/abs/1707.03388

The goal is not to declare dense attention universally better. It is to state
which long-range hypothesis each operator actually makes.

## 2. Bag, geometry, and attention notation

Let the sequence for slide i contain a class row and n_i patch rows:

```math
Z_i
=
\begin{bmatrix}
z_{i0}^{\mathsf T}\\
z_{i1}^{\mathsf T}\\
\vdots\\
z_{in_i}^{\mathsf T}
\end{bmatrix}
\in\mathbb R^{(n_i+1)\times d}.
```

The index zero is the class token. For one attention head, define

```math
Q_i=Z_iW_Q,
\qquad
K_i=Z_iW_K,
\qquad
V_i=Z_iW_V,
\qquad
W_Q,W_K,W_V\in\mathbb R^{d\times d_h}.
```

The score and attention matrices are

```math
S_i
=
\frac{Q_iK_i^{\mathsf T}}{\sqrt{d_h}},
\qquad
A_i
=
\mathrm{softmax}_{\mathrm{row}}(S_i).
```

The output is

```math
O_i=A_iV_i.
```

For an unmasked finite score matrix, each row has strictly positive
coefficients:

```math
(A_i)_{rs}>0,
\qquad
\sum_{s=0}^{n_i}(A_i)_{rs}=1.
```

This is the first source of confusion. Strict positivity means dense
mathematical support, not equal or useful influence.

Let a physical patch coordinate be c_ij and an imposed raster coordinate be
g_ij. A distance-based long-range statement must specify which coordinate is
being used:

```math
d_{\mathrm{phys}}(j,k)
=
\left\|c_{ij}-c_{ik}\right\|_2,
\qquad
d_{\mathrm{grid}}(j,k)
=
\left\|g_{ij}-g_{ik}\right\|_2.
```

Attention indices alone do not establish physical distance.

## 3. Mathematical support

### 3.1 Exact support

For a matrix A, define its support by

```math
\mathrm{supp}(A)
=
\{(r,s):A_{rs}\ne 0\}.
```

For unmasked softmax attention,

```math
\mathrm{supp}(A_i)
=
\{0,\ldots,n_i\}^2.
```

Every query can route value information from every key in one layer. The
direct computational graph has diameter one between any two token rows.

This statement does not imply that the row output is equally sensitive to every
input. The coefficient can be exponentially small:

```math
(A_i)_{rs}
=
\frac{\exp(S_{i,rs})}
\sum_t\exp(S_{i,rt)}.
```

If one score exceeds another by delta, their coefficient ratio is

```math
\frac{(A_i)_{rs}}{(A_i)_{ru}}
=
\exp\!\left(S_{i,rs}-S_{i,ru}\right)
=
\exp(\delta).
```

The denominator couples all keys in the row. A patch can therefore lose
coefficient mass not only because its own score falls, but because another
patch's score rises.

### 3.2 Masked support

Let M_i be an additive attention mask. Entries equal to negative infinity are
excluded:

```math
\widetilde S_{i,rs}
=
\begin{cases}
S_{i,rs},&M_{i,rs}=0,\\
-\infty,&M_{i,rs}=-\infty.
\end{cases}
```

Then

```math
\mathrm{supp}\!\left(
\mathrm{softmax}_{\mathrm{row}}(\widetilde S_i)
\right)
=
\{(r,s):M_{i,rs}=0\}.
```

A local mask creates a graph whose one-layer edges follow the prescribed
neighborhood. Repeated layers enlarge the reachable set through graph powers:

```math
\mathrm{Reach}_L
=
\mathrm{supp}\!\left(M_i+M_i^2+\cdots+M_i^L\right),
```

where the expression is interpreted as Boolean graph composition rather than
ordinary numeric addition when masks are binary.

For a connected local grid, depth can eventually produce global reach. For a
WSI bag with a very large diameter, the required depth may be too large for
optimization or memory.

## 4. Effective support

### 4.1 Threshold support

For a threshold epsilon greater than zero, define

```math
E_i(\varepsilon)
=
\{(r,s):(A_i)_{rs}>\varepsilon\}.
```

The cardinality depends on both epsilon and row:

```math
m_{i,r}(\varepsilon)
=
\left|\{s:(A_i)_{rs}>\varepsilon\}\right|.
```

Reporting a heatmap without epsilon hides this choice. A row with one large
coefficient and many tiny positive coefficients is mathematically dense but
effectively sparse for a finite-precision model.

### 4.2 Entropy support

The row entropy is

```math
\mathcal H_{i,r}
=
-\sum_{s=0}^{n_i}
(A_i)_{rs}\log\!\left((A_i)_{rs}+\epsilon_0\right),
```

where epsilon_0 only prevents taking a numerical logarithm of zero.

Normalize by the maximum entropy:

```math
\overline{\mathcal H}_{i,r}
=
\frac{\mathcal H_{i,r}}{\log(n_i+1)}.
```

Values near one indicate diffuse routing. Values near zero indicate
concentration. Entropy does not identify which tokens matter; it measures
dispersion.

### 4.3 Participation ratio

An alternative effective support size is

```math
\mathrm{PR}_{i,r}
=
\frac{1}{\sum_{s=0}^{n_i}(A_i)_{rs}^2}.
```

It equals n_i plus one for a uniform row and approaches one for a one-hot row.
The participation ratio is more sensitive to dominant coefficients than entropy.

### 4.4 Cumulative mass support

Sort row coefficients in descending order:

```math
a_{i,r,(1)}
\ge
a_{i,r,(2)}
\ge
\cdots
\ge
a_{i,r,(n_i+1)}.
```

Define the smallest k capturing mass tau:

```math
k_{i,r}(\tau)
=
\min\left\{
k:
\sum_{\ell=1}^{k}a_{i,r,(\ell)}
\ge\tau
\right\}.
```

This statistic distinguishes a row that spreads 95 percent of its mass across
many patches from one that places 95 percent on two patches. It still does not
measure causal influence.

## 5. Distance-conditioned support

To test long-range behavior, group coefficients by distance. For a query patch
j and key patch k, define a binning map B:

```math
\mathcal A_i(b)
=
\left\{
(A_i)_{jk}:
B\!\left(d_{\mathrm{phys}}(j,k)\right)=b
\right\}.
```

The mean routed mass in bin b is

```math
\mu_i(b)
=
\frac{1}{|\mathcal A_i(b)|}
\sum_{a\in\mathcal A_i(b)}a.
```

A globally connected attention matrix can still exhibit a rapidly decaying
distance-bin mean. Conversely, a small number of far-away bridge tokens can carry
high mass even when the average far-distance coefficient is small.

The relevant long-range statistic for a query row is the mass beyond a radius R:

```math
\mathrm{LRM}_{i,j}(R)
=
\sum_{k:d_{\mathrm{phys}}(j,k)>R}
(A_i)_{jk}.
```

For the class row, use class-to-patch coefficients:

```math
\mathrm{LRM}_{i,0}(R)
=
\sum_{k:d_{\mathrm{phys}}(0,k)>R}
(A_i)_{0k},
```

but only after defining a meaningful class-to-patch distance. If no physical
coordinate exists for the class row, a class-row distance is a modeling
convention rather than an observed quantity.

## 6. Influence is not attention

### 6.1 Direct value influence

Holding A fixed, the differential of one output row is

```math
dO_{i,r}
=
\sum_s(A_i)_{rs}\,dV_{i,s}.
```

This is the narrow setting in which attention coefficients quantify direct
linear value mixing.

### 6.2 Score-mediated influence

In the actual network, V and A both depend on the input. The full differential
is

```math
dO_{i,r}
=
\sum_s
(dA_i)_{rs}V_{i,s}
+
\sum_s
(A_i)_{rs}dV_{i,s}.
```

The first term changes routing, including the normalization denominator. The
second changes content. If a patch changes its key, it can alter the
coefficients assigned to every key in the same query row.

For a scalar prediction f_i, patch influence is represented by the Jacobian

```math
J_{ij}
=
\frac{\partial f_i}{\partial h_{ij}}.
```

The direct attention coefficient is only one factor inside this derivative.
With L layers and residual maps, the total derivative is a sum over paths:

```math
\frac{\partial f_i}{\partial h_{ij}}
=
\sum_{\pi\in\mathcal P(j\rightsquigarrow \mathrm{out})}
\prod_{e\in\pi}J_e.
```

The path set includes patch-patch edges, class edges, PPEG edges, residual
edges, normalization derivatives, and MLP derivatives.

### 6.3 Deletion influence

Let f_i^{(-j)} be the output after removing or masking patch j. The deletion
effect is

```math
\Delta_{ij}^{\mathrm{del}}
=
f_i-f_i^{(-j)}.
```

Deletion is nonlinear and redistributes attention over the remaining patches.
It is closer to a necessity test than a routing visualization, but it is not
automatically causal because the intervention may create an out-of-distribution
bag.

## 7. Dense attention and collective long-range signal

Suppose a query row contains m distant tokens, each with coefficient roughly
alpha and value direction v. Their aggregate contribution is

```math
\left\|
\sum_{\ell=1}^{m}\alpha v_\ell
\right\|_2.
```

If the vectors align, the norm can scale like m alpha. If they cancel, the
norm can be small even when total mass is large:

```math
\left\|
\sum_{\ell=1}^{m}\alpha v_\ell
\right\|_2
\le
\sum_{\ell=1}^{m}\alpha\|v_\ell\|_2.
```

Thresholding by individual coefficient can remove every member of a collectively
important group. This is why weak long-range edges cannot be dismissed solely
because each one is small.

A useful group intervention is to delete all tokens in a distance band:

```math
\Delta_i^{\mathrm{band}}(R_1,R_2)
=
f_i
-
f_i^{\left(
\{j:R_1<d_{\mathrm{phys}}(j,\cdot)\le R_2\}
\right)\mathrm{\ masked}}.
```

The group effect tests collective signal more directly than a single-patch
heatmap.

## 8. TransMIL Nyström attention

### 8.1 Exact factor dimensions

TransMIL reduces the cost of attention with a Nyström approximation. For
sequence length L_i plus one class row, let L denote the full sequence length
for one slide after square completion and class-token insertion. Let m be the
number of landmarks.

For the released implementation:

```math
d=512,
\qquad
H=8,
\qquad
d_h=64,
\qquad
m=256.
```

The exact attention score matrix for one head is L by L. Nyström attention
forms three softmax factors:

```math
A_{L,m}
=
\mathrm{softmax}_{\mathrm{row}}
\!\left(
\frac{QK_{\mathrm{land}}^{\mathsf T}}{\sqrt{d_h}}
\right),
```

```math
A_{m,m}
=
\mathrm{softmax}_{\mathrm{row}}
\!\left(
\frac{Q_{\mathrm{land}}K_{\mathrm{land}}^{\mathsf T}}{\sqrt{d_h}}
\right),
```

```math
A_{m,L}
=
\mathrm{softmax}_{\mathrm{row}}
\!\left(
\frac{Q_{\mathrm{land}}K^{\mathsf T}}{\sqrt{d_h}}
\right).
```

The approximate attention matrix is

```math
\widehat A
=
A_{L,m}
A_{m,m}^{+}
A_{m,L},
```

where the plus denotes a Moore-Penrose pseudoinverse or its iterative
approximation in code.

The output is

```math
\widehat O
=
\widehat A V.
```

### 8.2 Nominal support

If the three factors are dense, every entry of the product can be nonzero:

```math
\mathrm{supp}(\widehat A)
\subseteq
\{1,\ldots,L\}^2.
```

The approximation therefore does not impose a local sparse mask. It creates a
rank-constrained global interaction mediated by m landmark coordinates.

However, the pseudoinverse can contain negative entries. Therefore the
Nyström mixing matrix
is not generally a row-stochastic probability matrix even though the three
softmax factors individually are row-stochastic:

```math
\widehat A\mathbf 1
\ne
\mathbf 1
\quad\text{in general}.
```

Calling every entry of the product an attention probability is mathematically
misleading. It is an approximate mixing matrix.

### 8.3 Complexity

For fixed head width and landmark count, forming the three factors costs

```math
\mathcal O(Lmd+m^2d),
```

and multiplying them by values has comparable linear-in-L cost. Exact dense
attention costs

```math
\mathcal O(L^2d).
```

The phrase linear attention is conditional on m and d being treated as fixed.
If m grows with L, the asymptotic advantage changes. Memory, kernel
implementation, and the iterative pseudoinverse also contribute to the actual
runtime.

The official TransMIL code uses six Newton-Schulz-style pseudoinverse
iterations. Let P_0 be the initial approximate inverse and define a generic
iteration

```math
P_{t+1}
=
\frac{1}{2}P_t
\left(3I-A_{m,m}P_t\right).
```

The exact implementation's numerical behavior depends on initialization,
scaling, precision, and iteration count. A finite iteration is not the exact
Moore-Penrose inverse.

## 9. What the landmark bottleneck can lose

### 9.1 Low-rank channel

Ignoring nonlinearities and the pseudoinverse's signs, the approximation routes
messages through an m-dimensional landmark interface. The output is constrained
by the factorization

```math
\widehat O
=
A_{L,m}
\left(A_{m,m}^{+}A_{m,L}V\right).
```

The term in parentheses is a landmark-compressed value summary. A long-range
pattern that is not represented by the landmark interactions can be attenuated
even though the final matrix is numerically dense.

### 9.2 Error decomposition

Let A be exact attention. The output error satisfies

```math
\|AV-\widehat A V\|_F
\le
\|A-\widehat A\|_2\,\|V\|_F.
```

Thus a small operator-norm error is sufficient for all value matrices with
bounded Frobenius norm, while a large error can be hidden on one particular
bag if V lies in a favorable subspace.

For a particular class-row prediction, the relevant error is

```math
\left\|
e_0^{\mathsf T}(A-\widehat A)V
\right\|_2.
```

Global matrix approximation error and slide-level output error need not rank
models in the same way.

### 9.3 Landmark coverage

If landmarks are selected uniformly in token order, they may underrepresent
rare tissue modes. Let R be a rare subspace and let P_R be its projection. A
simple coverage diagnostic is

```math
\mathrm{Cov}_{R}
=
\frac{1}{m}
\sum_{\ell=1}^{m}
\left\|P_R z_{\ell}\right\|_2^2.
```

Low coverage does not prove failure, because the key and value maps can still
encode the rare mode indirectly. It does identify a plausible bottleneck.

## 10. Softmax temperature and support

Introduce temperature tau:

```math
A_i(\tau)
=
\mathrm{softmax}_{\mathrm{row}}
\!\left(S_i/\tau\right).
```

As tau approaches zero, rows concentrate on maximal scores:

```math
\lim_{\tau\to 0^+}
A_{i,rs}(\tau)
\in
\mathrm{ArgMaxRow}(S_i).
```

Ties can distribute mass among multiple maxima. As tau grows without bound,
the row approaches uniform:

```math
\lim_{\tau\to\infty}
A_{i,rs}(\tau)
=
\frac{1}{n_i+1}.
```

Temperature therefore changes effective support without changing the nominal
support of finite softmax. It also changes gradient scale:

```math
\frac{\partial A_{rs}}{\partial S_{ru}}
=
\frac{1}{\tau}A_{rs}
\left(\mathbf 1\{s=u\}-A_{ru}\right).
```

Low temperature can create sharper but more brittle routing. High temperature
can preserve broad context while diluting rare positives.

## 11. Sparsemax and entmax comparison

Sparsemax projects scores onto the probability simplex:

```math
\mathrm{sparsemax}(s)
=
\arg\min_{p\in\Delta^{n-1}}
\|p-s\|_2^2.
```

Its output can have exact zeros:

```math
\left|\mathrm{supp}(\mathrm{sparsemax}(s))\right|
<
n.
```

Entmax generalizes the sparsity level with a parameter alpha:

```math
\mathrm{entmax}_{\alpha}(s)
=
\arg\max_{p\in\Delta^{n-1}}
\left[
p^{\mathsf T}s
+
\mathcal H_{\alpha}(p)
\right].
```

Replacing softmax with a sparse normalizer changes the support at the operator
level, not merely the visualization threshold. This can improve interpretability
or reduce noise, but it makes long-range collective signal vulnerable to exact
zeroing.

The correct comparison is not dense versus sparse in the abstract. It is:

```math
\text{task signal}
\quad\text{versus}\quad
\text{support budget}
\quad\text{versus}\quad
\text{normalizer geometry}.
```

## 12. Local PPEG plus global attention

PPEG contributes a bounded local dependency set. For a grid location r, let its
7 by 7 neighborhood be the local stencil. One PPEG application gives

```math
\mathrm{Dep}_{\mathrm{PPEG}}(r)
\subseteq
\mathcal N_7(r)
\quad\text{within the same channel}.
```

Dense attention then gives

```math
\mathrm{Dep}_{\mathrm{SA}}(r)
=
\{0,\ldots,L-1\}
\quad\text{for an unmasked finite-score row}.
```

The two mechanisms are complementary:

- PPEG supplies a shared local inductive bias;
- attention supplies direct global reach;
- the class token supplies a slide-level state;
- Nyström reduces the cost of that global reach.

PPEG does not make attention local, and Nyström does not restore a physical
coordinate relation that preprocessing destroyed.

## 13. Long-range dependence as a permutation experiment

Let the bag be partitioned into local coordinate groups G_1 through G_R. Define
a permutation that preserves each group but changes the order of groups:

```math
\Pi_{\mathrm{group}}
=
\Pi_R\circ\cdots\circ\Pi_1,
\qquad
\Pi_r(G_r)=G_r.
```

If the model has no position-dependent operation and attention is exact, the
prediction should remain invariant:

```math
f(\Pi_{\mathrm{group}}H_i)=f(H_i).
```

If the architecture has PPEG, sequence positions, or coordinate-derived bias,
the prediction can change. The intervention isolates sensitivity to intergroup
order while approximately preserving local group contents.

For an actual physical-coordinate intervention, keep the coordinate map fixed
and permute features across far groups. The interpretation is different:

```math
f\!\left(\Pi_{\mathrm{far}}H_i;\Gamma_i\right)
-
f\!\left(H_i;\Gamma_i\right)
```

tests whether distant features are assigned to the correct positions, while
within-group structure is preserved.

## 14. Long-range dependence versus long-range shortcut

A prediction can change under a distant-group permutation for two opposite
reasons:

1. the model learned a useful relation between distant tissue regions;
2. the model learned an order, slide-size, or acquisition shortcut.

To distinguish them, compare:

```math
\Delta_{\mathrm{distance}}
=
\mathcal M(\text{correct physical arrangement})
-
\mathcal M(\text{distance-preserving null}),
```

and

```math
\Delta_{\mathrm{order}}
=
\mathcal M(\text{correct physical arrangement})
-
\mathcal M(\text{random group arrangement}).
```

The null should preserve bag size, patch multiset, tissue fraction, and local
group statistics. Otherwise the intervention changes too many variables.

## 15. Failure modes

### 15.1 Dense but diffuse attention

Every entry is nonzero, but row entropy is near its maximum. The model has
global reach in principle and little selective routing in practice. Rare signals
can be diluted.

### 15.2 Sparse but brittle attention

Temperature, sparsemax, or learned score scale can create a very small support.
A single artifact can displace the useful patch, and a collective distant signal
can be deleted by one threshold decision.

### 15.3 Landmark undercoverage

Nyström landmarks can underrepresent rare morphology, causing global messages to
be routed through an inadequate low-dimensional interface.

### 15.4 Approximation cancellation

The pseudoinverse can produce signed coefficients. A nominally large route can
be canceled by another factor, and the approximate matrix is not a probability
distribution that can be interpreted row by row.

### 15.5 Token-order geometry

If token order is not a physical raster, PPEG and any position-sensitive
attention can learn a synthetic long-range relation.

### 15.6 Class-row overinterpretation

The class row can use patch-patch paths, residual memory, and PPEG. Its direct
coefficients cannot certify that the model ignored or used every other patch.

### 15.7 Finite-precision support

Under float16 or aggressive kernel fusion, very small softmax coefficients may
underflow or be rounded. The deployed support can be smaller than the symbolic
support.

## 16. Sanity checks

### Check A: exact versus approximate attention

On small bags where exact attention fits, compare

```math
\varepsilon_{\mathrm{attn}}
=
\frac{\|AV-\widehat A V\|_F}
\{\|AV\|_F+\epsilon_0\}.
```

Report matrix error and final slide-logit error separately.

### Check B: row stochasticity

For exact softmax, verify

```math
\left\|A\mathbf 1-\mathbf 1\right\|_\infty
\approx 0.
```

For the Nyström product, report the same quantity rather than silently calling
the product a probability matrix.

### Check C: support curve

Plot effective support against epsilon, entropy, participation ratio, and
cumulative mass. One scalar heatmap threshold is inadequate.

### Check D: distance curve

Compute long-range mass as a function of physical and grid distance. A gap
between the curves indicates that imposed raster geometry and physical geometry
are not equivalent.

### Check E: landmark sweep

Evaluate m in a small sweep while holding all other settings fixed:

```math
m\in\{64,128,256,512\}.
```

If performance changes sharply, the task is sensitive to the landmark bottleneck
or its numerical approximation.

### Check F: temperature sweep

Measure performance, entropy, and deletion stability across tau. A sharper row
is not automatically a better explanation.

### Check G: group deletion

Delete or mask distant groups collectively. Compare the effect with the sum of
individual deletion effects:

```math
\Delta^{\mathrm{group}}
\ne
\sum_{j\in G}\Delta_j^{\mathrm{del}}
\quad\text{in general}.
```

The difference measures interaction and redundancy.

### Check H: coordinate shuffle

Preserve the patch multiset and shuffle only coordinates or only token order.
The two interventions diagnose whether the model relies on feature identity,
geometry, or the serialization convention.

## 17. C/R/G/S placement

| Component | Attention-support realization |
|---|---|
| Context operator | Dense row-softmax self-attention, optionally combined with PPEG; TransMIL approximates the dense map through a landmark factorization. |
| Readout operator | A class-token row followed by residual, normalization, and task head. |
| Geometry | Exact attention is geometry-free without positional inputs; PPEG makes support depend on the supplied raster; distance-conditioned tests must distinguish raster from physical distance. |
| Surviving statistic | A weighted and contextually transformed global state; weak edges may remain symbolically present, while landmark compression, thresholding, or normalization can remove useful collective signal. |

The decomposition is

```math
\text{token sequence}
\xrightarrow{\text{score kernel}}
\text{routing support}
\xrightarrow{\text{value mixing}}
\text{contextual states}
\xrightarrow{\text{class readout}}
\text{slide statistic}.
```

The key design axis is not simply whether attention is global. It is which
global interactions survive the score normalization, numerical approximation,
landmark interface, and final compression.

## 18. Bottom line

Exact unmasked softmax attention has dense symbolic support, but the useful
support of a WSI transformer is governed by score scale, entropy, value
directions, positional geometry, and the approximation used to make the
operator affordable.

TransMIL's Nyström module preserves a globally connected factorized route, not
the exact dense attention matrix. Its pseudoinverse product is a mixing
operator, not automatically a probability map. PPEG supplies local spatial
structure, while Nyström addresses quadratic cost; neither should be credited
with the other's effect.

The honest summary is:

```math
\boxed{
\text{symbolic reach}
\ne
\text{effective support}
\ne
\text{predictive influence}
\ne
\text{physical long-range dependence}
}
```

Any claim that a transformer captures long-range WSI context should report the
coordinate definition, support statistic, approximation error, and intervention
that separates useful distant dependence from order or acquisition shortcuts.

## References

- Shao et al., "TransMIL: Transformer based Correlated Multiple Instance
  Learning for Whole Slide Image Classification," NeurIPS 2021.
  https://arxiv.org/abs/2106.00908
- Martins and Astudillo, "From Softmax to Sparsemax: A Sparse Model of
  Attention and Multi-Label Classification," ICML 2016.
  https://arxiv.org/abs/1602.02068
- Peters et al., "Sparse Sequence-to-Sequence Models," ACL 2019.
  https://arxiv.org/abs/1707.03388
