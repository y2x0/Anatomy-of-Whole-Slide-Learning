# SCL-WC Support Feedback And Failure Modes

## Discrete Support

```math
I_i^{+,k}(\theta)
=
\mathrm{TopK}
\left(
A_i(\theta),k
\right).
```

The loss depends on both continuous parameters and selected indices:

```math
\mathcal{L}_{\mathrm{WSCL}}
\left(
\theta;
I^{+,k}(\theta),
I^{-,k}(\theta),
I^{h,k}(\theta)
\right).
```

Top-k selection is piecewise constant. Backpropagation treats current support
as fixed and does not differentiate through index switches.

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

Correct early support can bootstrap rare lesions. Incorrect support can
stabilize stain, adipose, pen marks, or scanner artifacts.

## Selection Recall

```math
r_i^{(k)}
=
\frac{|T_i\cap I_i|}{|T_i|}.
```

If:

```math
r_i^{(k)}=0,
```

the positive memory contains no true lesion evidence from that slide.

## False-Positive Mass

```math
\eta_i^{\mathrm{benign}}
=
\frac{
|\mathcal{P}_i^{\mathrm{benign}}|
}{M_i^{+}}.
```

Uniform positive averaging gives selected benign patches the same target
weight as lesion patches.

## Rare-Class Availability

For a class with fixed prevalence and batch size `B`:

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

Memory banks help when this quantity is near zero, but add staleness and
historical sampling bias.

## Failure Principle

Cross-slide contrast improves only the geometry of selected support. It cannot
directly correct lesions that never enter that support.
