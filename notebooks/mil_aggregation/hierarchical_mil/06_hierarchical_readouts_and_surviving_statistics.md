# Hierarchical Readouts And Surviving Statistics

## 1. Readout is a choice at every boundary

Let H^{(\ell)} be the states at level ell. A hierarchical model does not have
one readout; it has a sequence

```math
H^{(\ell+1)}
=
\mathcal R_{\ell}
\left(
H^{(\ell)},P^{(\ell)}
\right),
\qquad
\ell=0,\ldots,L-1,
```

followed by a slide readout

```math
z_i
=
\mathcal R_{\mathrm{slide}}
\left(
H_i^{(L)}
\right).
```

The information that survives is the composition of all these operators.

## 2. Additive and normalized coarsening

For hard assignments, additive transfer is

```math
u_b
=
\sum_{a:\pi(a)=b}h_a.
```

Mean transfer is

```math
u_b
=
\frac{1}{m_b}
\sum_{a:\pi(a)=b}h_a.
```

Additive transfer preserves a count-sensitive first moment; mean transfer
preserves a count-normalized first moment. If a parent-level classifier is
linear, sum transfer allows the parent contribution to scale with m_b while
mean transfer does not.

A hierarchical sum can still become nonlinear after coarse context:

```math
z
=
\rho_L
\left(
\left\{
\rho_{L-1}
\left(
\left\{
\cdots
\rho_0(\{h_a\}_{a\in C_b})
\cdots
\right\}
\right)
\right\}
\right).
```

The final statistic is therefore not simply the sum of all fine states.

## 3. Attention at multiple levels

Within-parent attention is

```math
u_b
=
\sum_{a:\pi(a)=b}
\alpha_{ab}h_a,
\qquad
\sum_{a:\pi(a)=b}\alpha_{ab}=1.
```

Parent-level attention is

```math
z
=
\sum_{b=1}^{n_L}\beta_b u_b,
\qquad
\sum_{b=1}^{n_L}\beta_b=1.
```

Expanding the composition gives an effective fine-unit weight

```math
z
=
\sum_a
\left(
\beta_{\pi(a)}\alpha_{a,\pi(a)}
\right)h_a.
```

The weight is a product of local and coarse routing. A patch can have a large
local alpha and still have little final influence if its parent has small beta.
Conversely, a moderate local patch can survive because its region receives high
coarse attention.

This is why a local attention heatmap is not a complete explanation of a
hierarchical prediction.

## 4. Sparse and prototype-like readouts

A top-k parent readout is

```math
S_k(H)
=
\mathrm{TopK}_{b}
\left(
q_b
\right),
\qquad
z
=
\mathrm{Agg}
\left(
\{u_b:b\in S_k(H)\}
\right).
```

This changes the surviving statistic from a weighted first moment to a selected
order statistic. MaxS DTFD is the limiting k=1 case at the fine-to-pseudo-bag
boundary; a region-level top-k operator is analogous but not identical because
its candidates are already summaries.

A prototype readout uses distances to learnable prototypes p_m:

```math
r_{bm}
=
\exp
\left(
-\frac{\|u_b-p_m\|_2^2}{2\sigma_m^2}
\right),
\qquad
z_m
=
\sum_b\beta_b r_{bm}.
```

The slide statistic is now a vector of soft occupancy or similarity masses. It
can retain distributional information that a single weighted mean discards, but
its reliability depends on the number and coverage of parent units.

## 5. Multiscale concatenation

A model can retain several levels explicitly:

```math
z
=
\left[
\mathrm{Readout}_0(H^{(0)})
\middle\Vert
\mathrm{Readout}_1(H^{(1)})
\middle\Vert
\cdots
\middle\Vert
\mathrm{Readout}_L(H^{(L)})
\right].
```

This reduces the irreversibility of a single coarse bottleneck, but it does not
make the hierarchy lossless. Each Readout ell is itself a compression, and
concatenating summaries does not recover within-parent configurations that were
never retained.

## 6. Task-specific survival readout

For a Cox-style head, the final scalar can be

```math
\eta_i
=
w^{\mathsf T}z_i+b.
```

For discrete hazards at horizons tau_k, the readout can produce K logits:

```math
\eta_{ik}
=
w_k^{\mathsf T}z_i+b_k,
\qquad
h_{ik}
=
\sigma(\eta_{ik}).
```

The hierarchy is upstream of the survival representation. A better region
summary can improve risk prediction, but the scalar Cox head still imposes a
proportional-hazards restriction on the final risk representation.

## 7. Surviving-statistic ledger

```text
Boundary operator                  Dominant surviving statistic
---------------------------------------------------------------------------
sum coarsening                     count-sensitive first moment
mean coarsening                    normalized first moment
attention coarsening               weighted first moment
top-k selection                    selected order statistics
prototype occupancy                soft distribution summary
multiscale concatenation           several task-dependent summaries
graph assignment plus sum          coarse statistic of fine graph states
```

The ledger should be reported with the model. "Hierarchical attention" alone
does not say what information reaches the task head.
