# Similarity, Presence, and Contribution

## 1. Three Maps

For patch `j`, prototype `k`, and class `c`, define

```math
\text{similarity: }s_{ijk},
\qquad
\text{presence: }u_{ik}=A_k(\{s_{ijk}\}_j),
\qquad
\text{credit: }\Gamma_{ikc}.
```

Even in a linear prototype head,

```math
F_{ic}=b_c+\sum_k w_{ck}u_{ik},
\qquad
\Gamma_{ikc}=w_{ck}u_{ik},
```

only credit is an additive decomposition of the logit. Presence is a readout;
similarity is a local relation.

## 2. Patch-Level Credit Is Not Automatic

Under max aggregation,

```math
u_{ik}=\max_j s_{ijk},
```

a unique winner permits the local decomposition

```math
\Gamma_{ijkc}
=w_{ck}s_{ijk}\mathbf 1\{j=j_{ik}^\star\}.
```

At ties this decomposition is nonunique. Under soft aggregation,

```math
u_{ik}=\sum_j a_{ijk}s_{ijk},
\qquad \sum_j a_{ijk}=1,
```

the product of output weight, attention weight, and similarity decomposes the
current forward pass. It does not equal deletion effect because removing patch
`j` can change
all normalized weights.

## 3. Nonlinear Heads

If

```math
F_{ic}=H_c(u_i)
```

is nonlinear, there is generally no canonical prototype credit satisfying

```math
F_{ic}=b_c+\sum_k\Gamma_{ikc}.
```

Gradient products, Integrated Gradients, Shapley values, and ablation effects
answer different local, path, coalition, and intervention questions. A
prototype visualization does not choose among them.

## 4. Necessity and Sufficiency

Let the first modified score below remove prototype block `k`, and let the
second retain only that block under a specified baseline. Then

```math
N_{ikc}=F_{ic}-F_{ic}^{(-k)},
\qquad
Q_{ikc}=F_{ic}^{(k)}-F_{ic}^{(0)}.
```

High similarity need imply neither high necessity nor sufficiency. These are
model-intervention quantities and require an explicit replacement semantics.
