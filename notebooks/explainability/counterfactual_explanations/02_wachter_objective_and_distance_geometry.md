# Wachter Objective and Distance Geometry

## 1. Penalized Target Search

Wachter et al. hold the trained model fixed and search for `x_prime` by
minimizing a target penalty plus distance:

```math
\mathcal L(x';x,y_{\star},\lambda)
=\lambda\big(f(x')-y_{\star}\big)^2
+d(x,x').
```

The constrained ideal

```math
\min_{x'}d(x,x')
\quad\text{subject to}\quad
f(x')=y_{\star}
```

is approximated by increasing `lambda` until the target tolerance is met. The
penalty parameter determines the tradeoff between target accuracy and closeness.

## 2. Robustly Scaled L1 Distance

For feature `k`, let

```math
\mathrm{MAD}_k
=\mathrm{median}_r
\left|x_{rk}-\mathrm{median}_s x_{sk}\right|.
```

The normalized Manhattan distance is

```math
d_{\mathrm{MAD}}(x,x')
=\sum_{k=1}^p
\frac{|x_k-x'_k|}{\mathrm{MAD}_k}.
```

This reduces domination by high-scale features and encourages sparse changes
through L1 geometry. It does not encode clinical difficulty unless inverse MAD
happens to align with intervention cost.

## 3. Metric Chooses the Explanation

For a linear boundary

```math
f(x)=\mathbf 1\{w^{\top}x+b\ge0\},
```

the nearest L2 counterfactual on the boundary is

```math
x^{\star}
=x-
\frac{w^{\top}x+b}{\|w\|_2^2}w.
```

Under weighted L1 distance, an optimum often changes a small number of features
with favorable boundary-effect-to-cost ratio. Thus L2 gives distributed motion
normal to the boundary, while L1 tends toward coordinate-sparse explanations.

## 4. Multiple Local Minima

Wachter et al. initialize optimization from different random points and retain
good minima. Diversity obtained this way is incidental: repeated runs can
collapse to the same basin, and there is no explicit repulsion among solutions.
DiCE turns diversity into part of the objective.

## 5. WSI Warning

Pixel L1 or L2 closeness is poorly aligned with histologic identity. Small
stain shifts can have large pixel cost, while imperceptible adversarial changes
can cross a classifier boundary. WSI counterfactuals require a domain-specific
state space and distance.

