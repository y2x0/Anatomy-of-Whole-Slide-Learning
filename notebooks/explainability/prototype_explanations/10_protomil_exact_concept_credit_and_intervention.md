# ProtoMIL Exact Concept Credit and Intervention

ProtoMIL's linear concept head yields an exact score decomposition unavailable
to many prototype systems.

## 1. Slide Forward Map

Let patch `j` have sparse activation vector

```math
h_{ij}=(h_{ij1},\ldots,h_{ijK})^{\top},
```

and attention weight `a_ij` satisfying

```math
a_{ij}\ge0,
\qquad
\sum_{j=1}^{n_i}a_{ij}=1.
```

The slide concept vector is

```math
z_i=\sum_{j=1}^{n_i}a_{ij}h_{ij},
\qquad
z_{ik}=\sum_j a_{ij}h_{ijk}.
```

For linear class head

```math
F_{ic}=b_c+w_c^{\top}z_i,
```

substitution gives

```math
F_{ic}
=b_c+\sum_{k=1}^K\sum_{j=1}^{n_i}
w_{ck}a_{ij}h_{ijk}.
```

## 2. Exact Credit at Two Resolutions

Concept-level and patch-concept credit are

```math
\kappa_{ikc}=w_{ck}z_{ik}
=\sum_jw_{ck}a_{ij}h_{ijk},
```

```math
\gamma_{ijkc}=w_{ck}a_{ij}h_{ijk}.
```

They satisfy exact completeness for the logit:

```math
F_{ic}-b_c
=\sum_k\kappa_{ikc}
=\sum_j\sum_k\gamma_{ijkc}.
```

This is algebraic score credit. Because attention depends on the full bag,
deleting a patch and recomputing attention need not change the score by

```math
\sum_k\gamma_{ijkc}.
```

## 3. Signed Explanations

Sparse activations are nonnegative, so the sign of concept credit is controlled
by `w_ck`. High concept presence can be evidence against a class:

```math
z_{ik}>0,
\qquad w_{ck}<0
\quad\Longrightarrow\quad
\kappa_{ikc}<0.
```

A top-activation display without classifier weights omits this distinction.

## 4. Concept Intervention

For a suspected spurious coordinate set `Q`, define

```math
h_{ijk}^{\mathrm{do}(Q=0)}
=\begin{cases}
0,&k\in Q,\\
h_{ijk},&k\notin Q.
\end{cases}
```

At a fixed trained head and fixed attention, the logit change is exactly

```math
F_{ic}-F_{ic}^{\mathrm{do}(Q=0)}
=\sum_{k\in Q}\kappa_{ikc}.
```

ProtoMIL's retraining-based removal is stronger: parameters and attention can
adapt, so its effect is a changed-model intervention rather than the fixed-model
identity above. It also requires an expert to identify the coordinate and a
second training run. The fact that the retrained model no longer uses a removed
coordinate is expected by construction; it does not establish that the
coordinate was causally spurious in the data-generating process. External
stress tests or domain-shift evaluation are needed to support that stronger
claim. The two interventions answer different questions.
