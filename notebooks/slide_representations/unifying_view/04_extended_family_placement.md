# External Structure And Representation Boundaries

The phrase "slide representation" is often used for several different objects:
patch features, spatial coordinates, a graph, a memory neighborhood, or a
pretrained latent vector. These should not be merged casually. The same patch
carrier can support different prediction objects depending on what is available
to the model at inference time.

## One Carrier, Several Objects

Start with tiled observations:

```math
X_i=\{(x_{ij},c_{ij})\}_{j=1}^{n_i},
\qquad
h_{ij}=E_\phi(x_{ij}).
```

The patch carrier can then be exposed in different ways:

```math
\begin{aligned}
\mathcal X_i^{\mathrm{set}}
&=\{h_{ij}\}_{j=1}^{n_i},\\
\mathcal X_i^{\mathrm{seq}}
&=(h_{i,\sigma_i(1)},\ldots,h_{i,\sigma_i(n_i)}),\\
\mathcal X_i^{\mathrm{graph}}
&=(V_i,E_i,H_i),\\
\mathcal X_i^{\mathrm{dist}}
&=\frac{1}{n_i}\sum_{j=1}^{n_i}\delta_{h_{ij}},\\
\mathcal X_i^{\mathrm{hier}}
&=(V_i^{(0:L)},\pi_i^{(0:L-1)},H_i^{(0:L)}).
\end{aligned}
```

The set and empirical distribution contain the same unordered patch
multiplicities. The sequence, graph, and hierarchy add structure that is not
recoverable from the empirical distribution alone.

## Native Object Versus External Context

Let a model use an object and an external structure:

```math
\widehat y_i
=
\mathcal H_\omega
\left[
\mathcal R_\psi
\left(
\mathcal C_\theta(\mathcal X_i;\mathcal M)
\right)
\right].
```

Here the external object may be coordinates, a memory bank, a prompt, or a
pretrained map. The distinction is:

```text
native representation:
    part of the object whose invariances and sufficiency are being claimed

external context:
    an additional object consulted by the predictor at training or inference

task readout:
    the statistic actually exposed to the head
```

Calling every input to the forward pass "the representation" hides where the
inductive bias enters.

## Coordinates: Object Or Context

Coordinates are part of the slide object when the model consumes:

```math
\mathcal X_i^{\mathrm{coord}}
=
\{(h_{ij},c_{ij})\}_{j=1}^{n_i}
```

and defines a transformation law under a coordinate change. They are only
context for a set model when they are used to construct an adjacency matrix:

```math
E_i=\mathrm{kNN}(c_{i1},\ldots,c_{in_i}),
\qquad
\mathcal C(H_i;E_i).
```

In the second case, coordinates determine information flow but need not survive
the final readout. A graph model can therefore use geometry without producing a
geometric slide embedding.

The distinction is testable. Hold H_i fixed and perturb coordinates:

```math
\Delta_{\mathrm{coord}}
=
\left\|
\mathcal F(H_i,c_i)
-
\mathcal F(H_i,c_i')
\right\|.
```

If the coordinate sensitivity is nonzero, the chosen model is
coordinate-sensitive.
That sensitivity may be intended spatial reasoning or an unstable preprocessing
shortcut.

## Sequence Order Is A Construction

An unordered patch set does not contain a canonical sequence. A sequence model
therefore needs an ordering map:

```math
\sigma_i
=
\Sigma(H_i,c_i,\xi_i),
\qquad
\mathcal X_i^{\mathrm{seq}}
=
(h_{i,\sigma_i(1)},\ldots,h_{i,\sigma_i(n_i)}).
```

If the ordering map is a deterministic raster or space-filling traversal, order
can be a proxy for geometry. If the ordering is random, the model is being
trained on randomly chosen trajectories. If the ordering depends on learned
features, order itself becomes task-dependent.

Two orderings of the same carrier can produce different outputs:

```math
\mathcal F_{\mathrm{seq}}(\mathcal X_i^{\mathrm{seq}})
\ne
\mathcal F_{\mathrm{seq}}(\mathcal X_i^{\mathrm{seq}}_\pi).
```

Sequence performance should therefore be reported with the ordering rule and
its stability under alternative valid traversals.

## Memory Is Not A Property Of One Slide

A retrieval system uses a query and an archive:

```math
\mathcal X_i^{\mathrm{retr}}
=
(z_i,\mathcal M),
\qquad
\mathcal M=\{(k_r,v_r)\}_{r=1}^{R}.
```

The same query embedding can yield different predictions when the archive
changes:

```math
\widehat y_i(\mathcal M)
\ne
\widehat y_i(\mathcal M').
```

Thus the prediction object is not the query embedding alone. It is the pair of
the query embedding and the archive
plus the retrieval rule. This creates two evaluation regimes:

```text
inductive:
    the archive is fixed before evaluating a held-out slide

transductive:
    the archive may include related cases or same-patient material
```

Without this distinction, retrieval can turn patient or slide leakage into
apparently strong representation quality.

## Foundation Features Are Not Automatically A Slide Representation

A patch foundation encoder gives:

```math
h_{ij}=E_{\phi_0}(x_{ij}).
```

This is a patch representation. A downstream WSI model still has to choose:

```math
z_i
=
\mathcal R_\psi
\left(
\mathcal C_\theta
\left(
\{h_{ij},c_{ij}\}_{j=1}^{n_i}
\right)
\right).
```

Only a pretrained slide encoder supplies a native map of the form:

```math
z_i^{(0)}
=
F_{\phi_0}^{\mathrm{slide}}
\left(
\{h_{ij},c_{ij}\}_{j=1}^{n_i}
\right).
```

Using a patch foundation model with mean or attention MIL does not make the
result a pretrained slide representation. It makes a task-specific readout
over pretrained patch features.

## Equivalence Under A Readout

For any representation map T, define:

```math
\mathcal X_i\sim_T\mathcal X_k
\quad\Longleftrightarrow\quad
T(\mathcal X_i)=T(\mathcal X_k).
```

The downstream head cannot distinguish members of the same equivalence class.
For a set or distribution statistic:

```math
\mu_i=\mu_k
\quad\Longrightarrow\quad
T(\mu_i)=T(\mu_k).
```

For a graph:

```math
(H_i,E_i)\sim_{\mathcal R\circ\mathcal C}(H_k,E_k)
\quad\Longleftrightarrow\quad
\mathcal R(\mathcal C(H_i,E_i))
=
\mathcal R(\mathcal C(H_k,E_k)).
```

The graph may distinguish layouts before readout and still collapse them after
pooling. Representation richness is therefore not the same as information
surviving the full forward map.

## Paper Placement Boundary

The following anchors illustrate different boundaries:

```text
HIPT:
    hierarchical slide tokens are part of the pretrained computation

Patch-GCN:
    coordinate graph is context before a global slide readout

PANTHER:
    prototype assignments define a distribution statistic over patches

Prov-GigaPath:
    a tile encoder is composed with a dedicated slide-level encoder

RetCCL and Yottixel:
    retrieval quality depends on an archive and a similarity geometry
```

These are not interchangeable uses of the word representation. Each paper
should state which object exists before the task head and which objects are
consulted only as external context.

## Reporting Rule

Every slide-representation paper should report:

```text
carrier:
    patches, cells, regions, or pretrained tokens

structure:
    set, order, graph, hierarchy, empirical measure, or memory query

construction:
    coordinates, segmentation, ordering, clustering, or learned topology

context:
    which entities can interact before readout

readout:
    exact statistic exposed to the task head

external state:
    memory bank, text prompts, pretrained encoder, or cohort-dependent object

invariance:
    which transformations leave the prediction unchanged
```

The core question is not whether a model has a large embedding. It is which
equivalences it imposes before the task is allowed to decide what matters.
