# Retrieval Memory and Nearest-Neighbor Semantics

## 1. Memory Object

Let a memory contain indexed representations and metadata:

```math
\mathcal M=\{(m_r,y_r,c_r)\}_{r=1}^{R}.
```

For query `q`, retrieve

```math
\mathcal N_K(q)
=\arg\min_{A\subseteq[R],|A|=K}
\sum_{r\in A}d(q,m_r).
```

The retrieval result depends on encoder geometry, metric, index, and cohort.

## 2. kNN Decision Rule

A class estimate can be weighted by similarity:

```math
\widehat p(y\mid q)
=\frac{\sum_{r\in\mathcal N_K(q)}
w(q,m_r)\mathbf 1\{y_r=y\}}
{\sum_{r\in\mathcal N_K(q)}w(q,m_r)}.
```

This is an explicit readout from memory, unlike a post-hoc nearest-neighbor
display attached to an unrelated classifier.

## 3. Retrieval Explanation Claim

A retrieved patch supports:

```text
the query is near this reference under the learned representation and metric.
```

It does not prove the reference shares causal pathology, nor that it is the
most influential item for a separate neural head.

## 4. Cohort Shift

If the memory distribution changes,

```math
p_{\mathrm{memory}}(H)\ne p_{\mathrm{deployment}}(H),
```

neighbor identity and label composition can change even when the query encoder
is frozen. Retrieval explanations require external-cohort stability checks.

