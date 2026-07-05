# Time-Conditioned WSI Readouts

Continuous-time survival permits the slide representation itself to depend on
time.

The generic form is:

```math
H_i=\{h_{ij}\}_{j=1}^{n_i},
\qquad
z_i(t)=\mathcal R_t(\mathcal C_t(H_i;G_i)),
```

then:

```math
\lambda_i(t)=\rho(g_\theta(t,z_i(t))).
```

This is strictly richer than:

```math
z_i=\mathcal R(\mathcal C(H_i;G_i)),
\qquad
\lambda_i(t)=\rho(g_\theta(t,z_i)).
```

because the morphology used for risk may change with time.

## Time-Conditioned Attention

Let \(q(t)\) be a time query:

```math
q(t)=\phi_t(t)\in\mathbb R^r.
```

Patch keys:

```math
k_{ij}=K h_{ij}.
```

Attention weights:

```math
a_{ij}(t)
=
\frac{\exp(q(t)^\top k_{ij})}
{\sum_{\ell=1}^{n_i}\exp(q(t)^\top k_{i\ell})}.
```

Readout:

```math
z_i(t)=\sum_{j=1}^{n_i}a_{ij}(t)Vh_{ij}.
```

Hazard:

```math
\lambda_i(t)
=
\operatorname{softplus}
\left[
u^\top z_i(t)+b(t)
\right].
```

Now attention is a function:

```math
(t,H_i)\mapsto a_i(t)\in\Delta^{n_i-1}.
```

## Likelihood Gradient Through Time Attention

The continuous-time loss is:

```math
\mathcal L_i
=
-\delta_i\log\lambda_i(X_i)
+
\int_0^{X_i}\lambda_i(u)\,du.
```

The gradient with respect to a patch embedding \(h_{ij}\) is:

```math
\nabla_{h_{ij}}\mathcal L_i
=
-
\delta_i
\nabla_{h_{ij}}\log\lambda_i(X_i)
+
\int_0^{X_i}
\nabla_{h_{ij}}\lambda_i(u)\,du.
```

This says:

```text
increase hazard at the observed event time
decrease accumulated hazard before the observed time
```

for patches that the time-conditioned readout uses.

For censored cases:

```math
\delta_i=0,
```

so:

```math
\nabla_{h_{ij}}\mathcal L_i
=
\int_0^{X_i}
\nabla_{h_{ij}}\lambda_i(u)\,du.
```

Censored slides only push down accumulated pre-censoring hazard.

## Time-Conditioned Prototype Hazards

Let:

```math
p_{im}
=
\frac{1}{n_i}\sum_jq_{ijm}
```

be prototype prevalence. Define:

```math
\lambda_i(t)
=
\operatorname{softplus}
\left[
b(t)
+
\sum_{m=1}^{M}p_{im}r_m(t)
\right].
```

Here \(r_m(t)\) is a prototype-specific time effect.

The derivative with respect to prototype prevalence is:

```math
\frac{\partial\lambda_i(t)}{\partial p_{im}}
=
\operatorname{softplus}'(\cdot)r_m(t).
```

So the survival loss gradient is:

```math
\frac{\partial\mathcal L_i}{\partial p_{im}}
=
-
\delta_i
\frac{\operatorname{softplus}'(\xi_i(X_i))r_m(X_i)}
{\lambda_i(X_i)}
+
\int_0^{X_i}
\operatorname{softplus}'(\xi_i(u))r_m(u)\,du,
```

where:

```math
\xi_i(t)=b(t)+\sum_mp_{im}r_m(t).
```

This exposes exactly how a prototype is rewarded or penalized across time.

## Time-Conditioned Graph Readout

Let graph node states be:

```math
\widetilde h_{ij}=\operatorname{GNN}(H_i,G_i)_j.
```

A time-conditioned graph readout can be:

```math
z_i(t)
=
\sum_j a_{ij}(t)\widetilde h_{ij},
```

where:

```math
a_{ij}(t)
=
\operatorname{softmax}_j(q(t)^\top K\widetilde h_{ij}).
```

Graph message passing defines spatial context. Time attention selects which
contextualized regions matter at horizon \(t\).

## Relation To Cox

Cox is recovered if:

```math
\lambda_i(t)
=
\lambda_0(t)\exp(\eta_i),
```

with:

```math
\eta_i=f(z_i)
```

and \(z_i\) does not depend on \(t\).

Taking logs:

```math
\log\lambda_i(t)
=
\log\lambda_0(t)+\eta_i.
```

A continuous WSI model violates proportional hazards when:

```math
\log\lambda_i(t)-\log\lambda_j(t)
```

depends on \(t\). Time-conditioned readout makes this natural.

## What Statistic Survives

Shared readout:

```math
z_i=\mathcal R(H_i)
```

survives as one slide statistic used at all times.

Time-conditioned readout:

```math
z_i(t)=\mathcal R_t(H_i)
```

survives as a curve in representation space:

```math
t\mapsto z_i(t).
```

Prototype time hazards survive as:

```math
\{p_{im},r_m(t)\}_{m=1}^M.
```

Graph time readouts survive as:

```math
\{a_{ij}(t),\widetilde h_{ij}\}_{j=1}^{n_i}.
```

## Dense Summary

```math
\begin{aligned}
z_i(t)&=\sum_j a_{ij}(t)Vh_{ij},\\
\lambda_i(t)&=\operatorname{softplus}(u^\top z_i(t)+b(t)),\\
\mathcal L_i
&=
-\delta_i\log\lambda_i(X_i)
+\int_0^{X_i}\lambda_i(u)\,du,\\
\nabla_{h_{ij}}\mathcal L_i
&=
-\delta_i\nabla_{h_{ij}}\log\lambda_i(X_i)
+\int_0^{X_i}\nabla_{h_{ij}}\lambda_i(u)\,du.
\end{aligned}
```

Continuous-time WSI models can make morphology time-indexed, but only if
\(\mathcal R\) or \(\mathcal C\) is allowed to depend on time.
