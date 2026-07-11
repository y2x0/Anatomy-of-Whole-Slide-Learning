# SCL-WC Support Feedback And Failure Modes

## Discrete Support Map

The attention parameters define selected indices:

```math
I_i^{+,k}(\theta)
=
\mathrm{TopK}
\left(
A_i(\theta),k
\right).
```

The WSCL loss is therefore:

```math
\mathcal{L}_{\mathrm{WSCL}}
\left(
\theta;
I^{+,k}(\theta),
I^{-,k}(\theta),
I^{h,k}(\theta)
\right).
```

Top-k selection is piecewise constant. Ordinary backpropagation treats the
current support as fixed and does not differentiate through index changes.

## Feedback Loop

```math
\text{attention}
\longrightarrow
\text{selected patches}
\longrightarrow
\text{contrastive gradients}
\longrightarrow
\text{refined features}
\longrightarrow
\text{attention}.
```

Correct early support can bootstrap rare lesions. Incorrect early support can
stabilize stain, adipose, pen marks, or scanner artifacts.

## Selection Recall

Let true positive patch set be `T_i` and selected top-k set be `I_i`. Recall is:

```math
r_i^{(k)}
=
\frac{|T_i\cap I_i|}{|T_i|}.
```

If:

```math
r_i^{(k)}=0,
```

the cross-slide positive memory contains no true lesion evidence from slide
`i`, despite using its positive slide label.

## Class-Conditional Benign Collision

Suppose selected positive bags contain benign patches with nonzero probability.
Their contribution to positive target mass is approximately:

```math
\eta_i^{\mathrm{benign}}
=
\frac{
\sum_{p\in\mathcal{P}_i^{\mathrm{benign}}}
1
}{M_i^{+}}.
```

Uniform positive averaging gives mislabeled benign patches the same target
weight as lesion patches.

## Hard-Negative Instability

Hard negatives receive high softmax mass when similar:

```math
q_{ih}
\propto
\exp
\left(
z_i^{\top}z_h/\tau
\right).
```

If a hard negative is actually a false negative, low temperature magnifies the
wrong repulsive gradient exponentially.

## Class-Imbalance Effect

For rare class `c`, expected same-class positive bags in a random batch of size
`B` are:

```math
\mathbb{E}
\left[
N_c^{+}
\mid
Y_i=c
\right]
=
\left(
B-1
\right)
\pi_c.
```

Memory banks are necessary when this quantity is near zero, but memory then
introduces staleness and historical sampling bias.

## Failure Principle

Cross-slide contrast improves only the geometry of the selected support. It
cannot directly correct lesions that never enter that support.
