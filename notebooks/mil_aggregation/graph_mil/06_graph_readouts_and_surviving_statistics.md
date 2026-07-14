# Graph Readouts and Surviving Statistics

Graph context and graph readout answer different questions:

```math
\mathcal C:
\text{which node states can depend on which relations?}
```

```math
\mathcal R:
\text{which statistic of the contextualized node field reaches the task head?}
```

A graph architecture is therefore:

```math
H_i^{(0)}
\xrightarrow{\mathcal C_{\theta}(\,\cdot\,,G_i)}
\widetilde H_i
\xrightarrow{\mathcal R_{\theta}}
z_i
\xrightarrow{\mathcal H_{\theta}}
\widehat y_i.
```

The same graph context can feed a mean, max, attention, typed, or hierarchical
readout. Those readouts define different representation objects even when the
node states are identical.

## 1. Common graph notation

For slide i, let:

```math
\widetilde H_i
=
\begin{bmatrix}
\widetilde h_{i1}^{\mathsf T}\\
\vdots\\
\widetilde h_{in_i}^{\mathsf T}
\end{bmatrix}
\in
\mathbb R^{n_i\times d}.
```

The node state can be raw:

```math
\widetilde h_{iv}
=
h_{iv}^{(0)},
```

or contextualized:

```math
\widetilde h_{iv}
=
\mathcal C_{\theta,v}
\left(
H_i^{(0)},A_i,E_i,\tau_i,B_i
\right).
```

The readout is a map:

```math
\mathcal R:
\left(
\mathbb R^d
\right)^{n_i}
\longrightarrow
\mathbb R^q.
```

If the node order is arbitrary, the graph-level readout should be invariant:

```math
\mathcal R(P_i\widetilde H_i)
=
\mathcal R(\widetilde H_i).
```

The graph can still be relation-sensitive because the contextualized states
depend on the correspondingly transformed relation object.

## 2. Sum readout

The graph sum is:

```math
z_i^{\mathrm{sum}}
=
\sum_{v=1}^{n_i}
\widetilde h_{iv}.
```

It is permutation invariant:

```math
z_i^{\mathrm{sum}}
\left(
\widetilde h_{i,\pi(1)},\ldots,\widetilde h_{i,\pi(n_i)}
\right)
=
z_i^{\mathrm{sum}}
\left(
\widetilde h_{i1},\ldots,\widetilde h_{in_i}
\right).
```

It is count-sensitive. Equal duplication gives:

```math
z^{\mathrm{sum}}
\left(
H_i\uplus H_i
\right)
=
2z^{\mathrm{sum}}(H_i).
```

If all nodes equal u:

```math
\widetilde h_{iv}=u
\quad
\forall v
\Longrightarrow
z_i^{\mathrm{sum}}
=
n_i u.
```

The sum can preserve total cellularity, total tissue burden, or exposure. It
can also encode patch count, segmentation density, or slide area.

## 3. Mean readout

The normalized graph mean is:

```math
z_i^{\mathrm{mean}}
=
\frac{1}{n_i}
\sum_{v=1}^{n_i}
\widetilde h_{iv}.
```

Define the empirical node measure:

```math
\widehat\mu_i
=
\frac{1}{n_i}
\sum_{v=1}^{n_i}
\delta_{\widetilde h_{iv}}.
```

Then the mean is its first moment:

```math
z_i^{\mathrm{mean}}
=
\int x\,d\widehat\mu_i(x).
```

Equal duplication is invisible:

```math
z^{\mathrm{mean}}
\left(
H_i\uplus H_i
\right)
=
z^{\mathrm{mean}}(H_i).
```

Sum and mean are related by:

```math
z_i^{\mathrm{sum}}
=
n_i z_i^{\mathrm{mean}}.
```

The distinction remains after graph context. If the graph changes node states
according to degree or topology, the mean is still normalized over the number
of contextualized nodes.

## 4. Max readout

Coordinatewise max is:

```math
\left[
z_i^{\mathrm{max}}
\right]_r
=
\max_{1\le v\le n_i}
\left[
\widetilde h_{iv}
\right]_r.
```

It is permutation invariant and replication invariant:

```math
z^{\mathrm{max}}(H_i\uplus H_i)
=
z^{\mathrm{max}}(H_i).
```

The maximizing node can vary by coordinate:

```math
v_r^\star
\in
\arg\max_v
[\widetilde h_{iv}]_r.
```

Therefore max does not necessarily select one graph node. It constructs a
coordinatewise extreme vector whose coordinates can come from different
locations.

Max preserves an upper-tail statistic but loses multiplicity and most of the
distribution below each maximum.

## 5. Global attention readout

Let a shared score map produce:

```math
s_{iv}
=
g_{\theta}
\left(
\widetilde h_{iv}
\right).
```

Normalize globally:

```math
\alpha_{iv}
=
\frac{
\exp(s_{iv})
}{
\sum_{u=1}^{n_i}\exp(s_{iu})
}.
```

The graph representation is:

```math
z_i^{\mathrm{attn}}
=
\sum_{v=1}^{n_i}
\alpha_{iv}\widetilde h_{iv}.
```

In matrix form:

```math
z_i^{\mathrm{attn}}
=
\widetilde H_i^{\mathsf T}\alpha_i.
```

This is a learned weighted first moment. It lies in the convex hull of the
contextualized node states:

```math
z_i^{\mathrm{attn}}
\in
\mathrm{conv}
\{\widetilde h_{iv}\}_{v=1}^{n_i}.
```

The score map is nodewise, but the normalized coefficient is globally
competitive:

```math
\frac{
\partial\alpha_{iv}
}{
\partial s_{iu}
}
=
\alpha_{iv}
\left(
\mathbf 1\{u=v\}
-
\alpha_{iu}
\right).
```

Increasing one node score changes the readout mass of every other node.

## 6. Local graph attention versus global readout attention

A graph context layer may use neighbor attention:

```math
\beta_{i,vu}
=
\frac{
\exp(e_{i,vu})
}{
\sum_{r\in\mathcal N_i(v)}
\exp(e_{i,vr})
}.
```

The node update is:

```math
\widetilde h_{iv}
=
\Phi
\left(
h_{iv},
\sum_{u\in\mathcal N_i(v)}
\beta_{i,vu}Vh_{iu}
\right).
```

Global readout attention is:

```math
\alpha_{iv}
=
\frac{
\exp(g(\widetilde h_{iv}))
}{
\sum_{u\in V_i}
\exp(g(\widetilde h_{iu}))
}.
```

The index domains are different:

```math
\beta_{i,vu}
:
u\in\mathcal N_i(v),
\qquad
\alpha_{iv}
:
v\in V_i.
```

The first is part of C. The second is part of R. A global heatmap made from
alpha does not show which neighbor messages created each contextualized state.

## 7. Attention deletion at graph readout

Let:

```math
w_{iv}
=
\exp(s_{iv}),
\qquad
S_i
=
\sum_{u=1}^{n_i}w_{iu},
\qquad
A_i
=
\sum_{u=1}^{n_i}w_{iu}\widetilde h_{iu}.
```

Then:

```math
z_i
=
\frac{A_i}{S_i}.
```

If node v is removed while the remaining contextualized states and scores stay
fixed:

```math
z_{i,-v}
=
\frac{
A_i-w_{iv}\widetilde h_{iv}
}{
S_i-w_{iv}
}.
```

The exact readout difference is:

```math
z_i-z_{i,-v}
=
\frac{
\alpha_{iv}
}{
1-\alpha_{iv}
}
\left(
\widetilde h_{iv}
-
z_{i,-v}
\right).
```

The full graph deletion generally recomputes context:

```math
\Delta_{iv}^{(q)}
=
q(G_i)
-
q(G_i\setminus v).
```

That intervention changes adjacency, degrees, top-k support, type rows, or
assignment transfers depending on the graph family. The readout identity is
not the full graph causal effect.

## 8. Typed readout

For fixed type vocabulary A, define:

```math
V_{ia}
=
\{v\in V_i:\tau_i(v)=a\}.
```

A per-type readout is:

```math
z_{ia}
=
\mathcal R_a
\left(
\{\widetilde h_{iv}:v\in V_{ia}\}
\right).
```

Stack the rows in a fixed type order:

```math
Z_i^{\mathrm{typed}}
=
\begin{bmatrix}
z_{ia_1}^{\mathsf T}\\
\vdots\\
z_{ia_{|\mathcal A|}}^{\mathsf T}
\end{bmatrix}.
```

The fixed row coordinate means:

```math
Z_{i,a,:}
=
\text{summary for teacher or semantic type }a.
```

This preserves semantic type separation. It does not preserve within-type
spatial arrangement, node identities, or teacher uncertainty unless explicitly
included.

## 9. Typed mean, sum, and attention

Typed mean:

```math
z_{ia}^{\mathrm{mean}}
=
\frac{1}{|V_{ia}|}
\sum_{v\in V_{ia}}
\widetilde h_{iv}.
```

Typed sum:

```math
z_{ia}^{\mathrm{sum}}
=
\sum_{v\in V_{ia}}
\widetilde h_{iv}.
```

Typed attention:

```math
\alpha_{iv|a}
=
\frac{
\exp(s_{iv|a})
}{
\sum_{u\in V_{ia}}\exp(s_{iu|a})
},
```

```math
z_{ia}^{\mathrm{attn}}
=
\sum_{v\in V_{ia}}
\alpha_{iv|a}\widetilde h_{iv}.
```

The count of type a is:

```math
n_{ia}
=
|V_{ia}|.
```

Normalized typed mean discards count unless the type count is concatenated:

```math
z_i^{\mathrm{typed+count}}
=
\left[
Z_i^{\mathrm{typed}}
\middle\Vert
(n_{ia})_{a\in\mathcal A}
\right].
```

An absent type needs a mask:

```math
m_{ia}
=
\mathbf 1\{n_{ia}>0\}.
```

An all-zero row without a mask is ambiguous.

## 10. HACT hierarchical readout

Let cell states after cell context be:

```math
\widetilde H_{\mathrm C}
\in
\mathbb R^{N_{\mathrm C}\times d_{\mathrm C}}.
```

The assignment transfer is:

```math
U_{\mathrm C\to T}
=
B^{\mathsf T}\widetilde H_{\mathrm C}.
```

The tissue initialization is:

```math
H_{\mathrm T}^{(0)}
=
\left[
H_{\mathrm T}
\middle\Vert
U_{\mathrm C\to T}
\right].
```

After tissue graph context:

```math
\widetilde H_{\mathrm T}
=
\mathcal C_{\mathrm T}
\left(
H_{\mathrm T}^{(0)},A_{\mathrm T}
\right).
```

The tissue sum is:

```math
z_{\mathrm{HACT}}
=
\sum_{b=1}^{N_{\mathrm T}}
\widetilde h_{\mathrm T,b}.
```

In a linearized model, let the two effective propagation operators include the
learned cell- and tissue-level message transforms:

```math
z_{\mathrm{HACT}}
=
\mathbf 1^{\mathsf T}
\widetilde A_{\mathrm T}
\left[
H_{\mathrm T}
\middle\Vert
B^{\mathsf T}
\widetilde A_{\mathrm C}H_{\mathrm C}
\right].
```

This is a coarse-scale sum of fine-scale contextual sums. It is not a flat
first moment over cells.

## 11. Assignment nullspace and readout collision

If:

```math
B^{\mathsf T}\Delta=0,
```

then:

```math
B^{\mathsf T}(H_{\mathrm C}+\Delta)
=
B^{\mathsf T}H_{\mathrm C}.
```

If the tissue graph and original tissue features remain fixed, the final graph
readout collides:

```math
z_{\mathrm{HACT}}(H_{\mathrm C}+\Delta)
=
z_{\mathrm{HACT}}(H_{\mathrm C}).
```

The nullspace is a precise information-loss statement. Fine-scale
within-region contrasts with zero sum are not available to later tissue context.

## 12. WiKG mean and max readouts

WiKG updates head states:

```math
h_i^{+}
=
\sigma_1
\left(
W_1(h_i+h_{\mathcal N(i)})
\right)
+
\sigma_2
\left(
W_2(h_i\odot h_{\mathcal N(i)})
\right).
```

Mean readout:

```math
z_{\mathrm{WiKG}}^{\mathrm{mean}}
=
\frac{1}{N}
\sum_{i=1}^{N}h_i^{+}.
```

Max readout:

```math
[z_{\mathrm{WiKG}}^{\mathrm{max}}]_r
=
\max_i[h_i^{+}]_r.
```

The dynamic topology affects both readouts through the updated head states. The readout itself
does not preserve the complete directed support or edge embeddings.

## 13. Patch-GCN global attention

Patch-GCN produces contextualized nodes:

```math
\widetilde h_v
=
\mathcal C_{\mathrm{PatchGCN}}
\left(
H,A_{\mathrm{coord}}
\right)_v.
```

Global attention is:

```math
b_v
=
\frac{
\exp(g(\widetilde h_v))
}{
\sum_u\exp(g(\widetilde h_u))
}.
```

The WSI representation is:

```math
z_{\mathrm{PatchGCN}}
=
\sum_vb_v\widetilde h_v.
```

The surviving statistic is a weighted first moment of spatially contextualized
nodes. It is not a weighted first moment of isolated patch features.

## 14. HEAT PL readout

HEAT produces typed contextualized states:

```math
\widetilde h_v
=
\mathcal C_{\mathrm{HEAT}}
\left(
H,A,\tau,\phi
\right)_v.
```

PL rows are:

```math
S_a
=
\mathcal R_a
\left(
\{\widetilde h_v:\tau(v)=a\}
\right).
```

The final graph representation is:

```math
z_{\mathrm{HEAT}}
=
\mathcal R_{\mathrm{graph}}
\left(
S_{a_1},\ldots,S_{a_{|\mathcal A|}}
\right).
```

The type coordinate system survives; within-type arrangements do not unless
the per-type readout explicitly preserves them.

## 15. Graph readout collisions

A collision is:

```math
G\ne G'
\quad\text{but}\quad
\mathcal R(\mathcal C(G))
=
\mathcal R(\mathcal C(G')).
```

Once the representation collides, any downstream head H must produce the same
output:

```math
z(G)=z(G')
\Longrightarrow
\mathcal H(z(G))=\mathcal H(z(G')).
```

### Mean collision

```math
\frac{1}{N}
\sum_v\widetilde h_v
=
\frac{1}{N'}
\sum_u\widetilde h_u'.
```

Different distributions, tails, and graph topologies can collide.

### Sum collision

```math
\sum_v\widetilde h_v
=
\sum_u\widetilde h_u'.
```

Different node counts and average states can collide if their total vector agrees.

### Max collision

```math
\max_v\widetilde h_{v,r}
=
\max_u\widetilde h_{u,r}'
\quad
\forall r.
```

Multiplicity and all submaximal states are hidden.

### Attention collision

```math
\sum_va_v\widetilde h_v
=
\sum_ua_u'\widetilde h_u'.
```

Different attention distributions can yield the same weighted mean.

### Typed collision

```math
S_a(G)
=
S_a(G')
\quad
\forall a
```

implies a graph-level row readout cannot distinguish the graphs.

## 16. Count and duplication tests

Duplicate every node and its incident relation pattern. For sum:

```math
z^{\mathrm{sum}}(G\uplus G)
=
2z^{\mathrm{sum}}(G)
```

under identical contextualized states.

For mean:

```math
z^{\mathrm{mean}}(G\uplus G)
=
z^{\mathrm{mean}}(G)
```

only when duplication also preserves the context operator's node states and
normalization.

For attention, duplicating a node changes the softmax denominator:

```math
\alpha_j'
=
\frac{\exp(s_j)}
{\exp(s_j)+\exp(s_j)+\sum_{u\ne j}\exp(s_u)}.
```

The duplicated node splits mass with its copy, so the representation need not
remain unchanged.

For HACT, duplicating a cell inside one tissue region changes the assignment
sum but not a mean transfer. For HEAT, duplicating a typed node can change both
feature-kNN support and type-row normalization.

## 17. Tail and witness tests

Insert one extreme contextualized node:

```math
G'
=
G\cup\{u_{\mathrm{extreme}}\}.
```

Mean perturbation:

```math
z^{\mathrm{mean}}(G')-z^{\mathrm{mean}}(G)
=
\frac{
u_{\mathrm{extreme}}-z^{\mathrm{mean}}(G)
}{
N+1
}.
```

Max perturbation is coordinate-dependent:

```math
[z^{\mathrm{max}}(G')]_r
=
\max
\left(
[z^{\mathrm{max}}(G)]_r,
[u_{\mathrm{extreme}}]_r
\right).
```

Attention perturbation depends on both score and value:

```math
z^{\mathrm{attn}}(G')
=
\frac{
\sum_v\exp(s_v)v_v+\exp(s_{\mathrm{extreme}})u_{\mathrm{extreme}}
}{
\sum_v\exp(s_v)+\exp(s_{\mathrm{extreme}})
}.
```

The result can be small if the new node receives low score, even if its value
is extreme.

## 18. Distributed-burden tests

If the label depends on the fraction of positive nodes:

```math
Y
\approx
\mathbf 1
\left\{
\frac{1}{N}
\sum_vZ_v>\tau
\right\},
```

mean or count-aware typed readout is naturally aligned.

If the label depends on the existence of one positive node:

```math
Y
=
\mathbf 1
\left\{
\sum_vZ_v>0
\right\},
```

max, high quantile, or sparse attention may retain a witness more directly.

Graph context can change the meaning of Z-v by contextualizing node states. A
max over contextualized nodes is not the same as a max over isolated patch
scores.

## 19. Task-head dependence

The same z can feed different heads:

```math
\eta_i^{\mathrm{Cox}}
=
w_{\mathrm{Cox}}^{\mathsf T}z_i,
```

```math
\widehat p_i
=
\mathrm{softmax}
\left(
W_{\mathrm{cls}}z_i+b_{\mathrm{cls}}
\right),
```

```math
\widehat h_i^{(k)}
=
\sigma
\left(
w_k^{\mathsf T}z_i+b_k
\right).
```

The relevant target-specific score differs:

```math
q
\in
\{
\eta_i^{\mathrm{Cox}},
\ell_{i,c}^{\mathrm{cls}},
\ell_i^{(k)}
\}.
```

A graph attention coefficient is not a universal explanation for all heads.
A deletion score must recompute the selected task output.

## 20. Signed graph readout credit

For a locally linear head:

```math
q_i
=
w^{\mathsf T}z_i+b.
```

For attention readout:

```math
q_i
=
\sum_v
\alpha_{iv}
w^{\mathsf T}\widetilde h_{iv}
+b.
```

The signed additive term is:

```math
r_{iv}
=
\alpha_{iv}
w^{\mathsf T}\widetilde h_{iv}.
```

The attention coefficient is nonnegative:

```math
\alpha_{iv}\ge0,
```

but the signed term can be negative. A high-weight contextualized node can oppose the
target direction.

For graph context, the full node deletion effect is:

```math
\Delta_{iv}^{(q)}
=
q(G_i)-q(G_i\setminus v).
```

The difference between the signed term and the deletion effect measures graph interaction,
normalization, and topology effects.

## 21. Graph readout matrix

| Method | C output | R | Surviving statistic | Main information loss |
| --- | --- | --- | --- | --- |
| Patch-GCN | multiscale coordinate-contextualized nodes | global attention | weighted first moment | graph geometry and within-node message decomposition |
| HEAT | typed edge-conditioned node states | PL rows then graph readout | type-conditioned summaries | within-type arrangement and teacher uncertainty |
| WiKG | head states after directed KAA and dual interaction | mean or max | first moment or coordinatewise extrema | selected support and edge triples |
| HACT | tissue states initialized from cell-context sums | tissue sum or layerwise sums | hierarchical coarse statistic | cell configurations in assignment nullspace |
| generic graph MIL | relation-contextualized node field | sum, mean, max, attention | chosen global statistic | determined by R |

## 22. Complexity and representation budget

For a sparse graph with E edges and hidden width d, message passing is roughly:

```math
\mathcal O(|E|d^2)
```

up to layer and projection factors.

Global sum or mean is:

```math
\mathcal O(Nd).
```

Global attention is:

```math
\mathcal O(Nd)
```

plus score-network cost.

A dense graph context can be:

```math
\mathcal O(N^2d)
```

or worse with full pairwise projections. WiKG's dense candidate head-tail score
can have quadratic construction cost even though only N K edges survive.

HACT's cell-to-tissue transfer is:

```math
\mathcal O(\mathrm{nnz}(B)d).
```

The representation budget is not only compute. Every readout also sets the
number of statistics sent to the head:

```math
\text{node field of size }Nd
\longrightarrow
\text{graph vector of size }q.
```

## 23. C/R/G/S placement

| Readout | C | R | G | S |
| --- | --- | --- | --- | --- |
| graph sum | any graph context | count-sensitive sum | graph support enters through contextual states | task head sees total contextual mass |
| graph mean | any graph context | normalized first moment | relation support affects node states | task head sees average contextual state |
| graph max | any graph context | coordinatewise upper tail | relation support affects extrema | task head sees witnesses/extremes |
| graph attention | any graph context | learned weighted first moment | relation support affects values and scores | task target defines only downstream credit |
| HEAT PL | typed edge context | fixed type rows then graph readout | feature graph, types, edge attributes | teacher types are generated attributes |
| HACT | two graph contexts plus assignment | coarse hierarchy readout | cell graph, tissue graph, B | classification or extended task head |

The same R does not have the same semantics if C changes.

## 24. Sanity-check suite

### Permutation check

```math
f(PH,PAP^{\mathsf T})
=
f(H,A).
```

### Relation intervention

Hold H fixed and change A:

```math
f(H,A)
\ne
f(H,A')
```

when the model genuinely uses graph geometry.

### Readout ablation

Hold contextualized node states fixed and compare:

```math
z^{\mathrm{mean}},
\qquad
z^{\mathrm{max}},
\qquad
z^{\mathrm{attn}}.
```

This isolates R from C.

### Assignment nullspace

For HACT:

```math
B^{\mathsf T}\Delta=0
\Longrightarrow
z_{\mathrm{HACT}}(H+\Delta)
=
z_{\mathrm{HACT}}(H)
```

when downstream inputs are otherwise unchanged.

### Type shuffle

For HEAT:

```math
z(\tau)
\ne
z(\tau_{\mathrm{shuffle}})
```

in general. This tests whether the semantic type map affects C or R.

### Directed reverse-edge check

For WiKG:

```math
A_{ij}
\ne
A_{ji}
```

is allowed and should not be silently symmetrized.

### Context recomputation deletion

Compare fixed-node deletion and graph-rebuilt deletion. Their difference is
topology-mediated influence.

## 25. Bottom line

The graph readout is the final information bottleneck:

```math
\boxed{
\text{graph object}
\longrightarrow
\text{contextualized node field}
\longrightarrow
\text{surviving statistic}
\longrightarrow
\text{task head}
}
```

The principal operators retain different information:

```math
\begin{aligned}
z^{\mathrm{sum}}
&=
\sum_v\widetilde h_v,
\\
z^{\mathrm{mean}}
&=
\frac{1}{N}\sum_v\widetilde h_v,
\\
z^{\mathrm{max}}_r
&=
\max_v[\widetilde h_v]_r,
\\
z^{\mathrm{attn}}
&=
\sum_v\alpha_v\widetilde h_v,
\\
Z^{\mathrm{typed}}
&=
\left[
\mathcal R_a(\widetilde H_a)
\right]_{a\in\mathcal A},
\\
z^{\mathrm{HACT}}
&=
\mathcal R_{\mathrm T}
\left(
\mathcal C_{\mathrm T}
\left(
H_{\mathrm T}
\Vert
B^{\mathsf T}\mathcal C_{\mathrm C}(H_{\mathrm C})
\right)
\right).
\end{aligned}
```

A graph model is not fully described by its message-passing layer. The readout
determines the collision class that the task head must tolerate.

The C/R/G/S summary is:

```math
\boxed{
\text{context determines available relations}
+
\text{readout determines surviving statistic}
+
\text{geometry determines topology}
+
\text{supervision selects useful collisions}
}
```

The correct question is:

```math
\text{what information about the contextualized graph is still present after R?}
```