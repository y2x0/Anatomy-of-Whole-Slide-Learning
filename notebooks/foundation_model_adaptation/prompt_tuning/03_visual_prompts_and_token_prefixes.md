# Visual Prompts And Token Prefixes

Visual prompt tuning changes the input to a frozen visual encoder.

For a patch or tile image $x$, a visual prompt can be:

```math
\widetilde x
=
x+\delta_\eta,
```

or a learned prefix token sequence:

```math
\widetilde T
=
(p_1,\ldots,p_m,t_1,\ldots,t_n).
```

The frozen encoder produces:

```math
h_\eta
=
f_{\phi_0}(\widetilde x)
```

or:

```math
h_\eta
=
F_{\phi_0}(\widetilde T).
```

## Task Information

Task labels train:

```math
\eta,
```

not the main encoder:

```math
\nabla_{\phi_0}\mathcal{L}
=
0.
```

The prompt changes activations indirectly through attention to prefix tokens.

## Prefix Attention View

In a transformer layer, tokens attend to:

```math
[P_\eta;T].
```

If $Q,K,V$ are frozen, prompt tokens change the attention output by adding new
keys and values:

```math
\mathrm{Attn}(T,[P_\eta;T],[P_\eta;T]).
```

Thus task information enters the context operator without changing $Q,K,V$.

## Dense Summary

Visual and prefix prompts are input-space or token-space adaptation:

```math
T
\to
[P_\eta;T]
\to
F_{\phi_0}([P_\eta;T]).
```

They can steer a frozen network, but they cannot directly rewrite its internal
feature maps.
