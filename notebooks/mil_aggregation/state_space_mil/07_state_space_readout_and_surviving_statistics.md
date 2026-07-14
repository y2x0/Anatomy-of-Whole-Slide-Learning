# State-Space Readout and Surviving Statistics

Let an ordered patch sequence enter a state-space context operator:

```math
U_i
=
\begin{bmatrix}
u_{i1}^{\mathsf T}\\
\vdots\\
u_{iL_i}^{\mathsf T}
\end{bmatrix},
\qquad
Y_i
=
\mathcal C_{\theta}^{\mathrm{SSM}}(U_i)
=
\begin{bmatrix}
y_{i1}^{\mathsf T}\\
\vdots\\
y_{iL_i}^{\mathsf T}
\end{bmatrix}
\in
\mathbb R^{L_i\times d}.
```

The readout is a separate operator:

```math
z_i
=
\mathcal R_\theta(Y_i,M_i)
\in
\mathbb R^q.
```

Here M-i is a valid-token mask when padding is present. The state-space scan
makes Y order-conditioned; R decides which part of that trajectory reaches the
task head.

The central distinction is:

```math
\boxed{
\text{C creates a trajectory}
\qquad
\text{R chooses the surviving trajectory statistic}
}
```

## 1. Source anchors

The paper-specific readout assignments are:

| Method | Source note | Readout used here |
| --- | --- | --- |
| S4MIL | [S4MIL structured readout](03_s4mil_structured_state_space_readout.md) | coordinatewise max of S4D contextualized outputs |
| MambaMIL | [MambaMIL vanilla scan](05_mambamil_vanilla_scan_and_mil_head.md) | released-code scalar attention weighted mean |
| SR-Mamba | [SR-Mamba reordering](06_srmamba_sequence_reordering_operator.md) | fused multi-order trajectory followed by a downstream readout |

A strict final-state or mean readout is a mathematical comparison operator. It
should not be reported as the paper's readout unless the implementation uses it.

## 2. Coordinatewise max

S4MIL uses:

```math
[z_i^{\mathrm{max}}]_r
=
\max_{1\le t\le L_i}
[y_{it}]_r.
```

The maximizing position can vary by channel:

```math
t_{ir}^\star
\in
\underset{1\le t\le L_i}{\arg\max}
[y_{it}]_r.
```

Therefore the vector may be assembled from different tokens:

```math
z_i^{\mathrm{max}}
=
\begin{bmatrix}
[y_{i,t_{i1}^\star}]_1\\
\vdots\\
[y_{i,t_{id}^\star}]_d
\end{bmatrix}.
```

It need not equal any actual contextualized token y-it.

Max is invariant to output-row permutation:

```math
\mathcal R_{\mathrm{max}}(PY)
=
\mathcal R_{\mathrm{max}}(Y).
```

It is not invariant to input permutation before the scan:

```math
\mathcal R_{\mathrm{max}}
\left(
\mathcal C_{\mathrm{SSM}}(PU)
\right)
\ne
\mathcal R_{\mathrm{max}}
\left(
\mathcal C_{\mathrm{SSM}}(U)
\right)
```

in general.

Max retains an upper-tail statistic but discards:

```text
response multiplicity
response position after the scan
submaximal evidence
co-occurrence of channel maxima
signed evidence below the maximum
```

For a sparse witness target, this can be an appropriate inductive bias. For a
burden target, it is usually insufficient.

## 3. Mean readout

The trajectory mean is:

```math
z_i^{\mathrm{mean}}
=
\frac{1}{L_i}
\sum_{t=1}^{L_i}y_{it}.
```

With a mask:

```math
z_i^{\mathrm{mean}}
=
\frac{1}{\sum_t m_{it}}
\sum_{t=1}^{L_i}
m_{it}y_{it}.
```

It is the first moment of the empirical output measure:

```math
\widehat\mu_i^Y
=
\frac{1}{L_i}
\sum_{t=1}^{L_i}
\delta_{y_{it}},
\qquad
z_i^{\mathrm{mean}}
=
\int y\,d\widehat\mu_i^Y(y).
```

Mean is invariant to duplicating the complete trajectory when the contextual
states are held fixed:

```math
z^{\mathrm{mean}}(Y\uplus Y)
=
z^{\mathrm{mean}}(Y).
```

But duplicating an input token before the SSM changes later prefix states, so:

```math
\mathcal R_{\mathrm{mean}}
\left(
\mathcal C_{\mathrm{SSM}}(U\uplus u_j)
\right)
\ne
\mathcal R_{\mathrm{mean}}
\left(
\mathcal C_{\mathrm{SSM}}(U)
\right)
```

in general.

Mean therefore removes multiplicity only after C. It does not make the whole
state-space model a set function.

## 4. Attention weighted mean

The released MambaMIL head scores contextualized tokens:

```math
s_{it}
=
w_2^{\mathsf T}
\tanh
\left(
W_1y_{it}+b_1
\right)
+
b_2.
```

Normalize over valid tokens:

```math
\alpha_{it}
=
\frac{
m_{it}\exp(s_{it})
}{
\sum_{u=1}^{L_i}m_{iu}\exp(s_{iu})
}.
```

The slide vector is:

```math
z_i^{\mathrm{attn}}
=
\sum_{t=1}^{L_i}
\alpha_{it}y_{it}.
```

This is a learned weighted first moment of order-conditioned states:

```math
z_i^{\mathrm{attn}}
=
\int y\,d\widehat\mu_{i,\alpha}^Y(y).
```

Because alpha-it are nonnegative and sum to one:

```math
z_i^{\mathrm{attn}}
\in
\mathrm{conv}
\{y_{it}:m_{it}=1\}.
```

The weight is a representation coefficient, not a posterior probability of
instance positivity:

```math
\alpha_{it}
\ne
\Pr(Z_{it}=1\mid Y_i,\mathcal B_i)
```

unless an additional probabilistic model defines and calibrates that
interpretation.

## 5. Attention normalization and global competition

The softmax derivative is:

```math
\frac{\partial\alpha_{it}}{\partial s_{iu}}
=
\alpha_{it}
\left(
\mathbf 1\{t=u\}
-
\alpha_{iu}
\right).
```

Increasing one token score changes every other coefficient. For a perturbation
of one state y-ik, the readout derivative has two paths:

```math
\frac{\partial z_i}{\partial y_{ik}}
=
\alpha_{ik}I_d
+
\sum_{t=1}^{L_i}
y_{it}
\frac{\partial\alpha_{it}}{\partial y_{ik}}.
```

The first term is the value path. The second is the score-normalization path.
A high coefficient can be accompanied by a small task effect if its value lies
near the current weighted mean.

For a scalar direction w:

```math
q_i
=
w^{\mathsf T}z_i+b,
```

the signed additive term is:

```math
r_{it}
=
\alpha_{it}w^{\mathsf T}y_{it}.
```

It can be negative even when alpha-it is large.

## 6. Exact fixed-state attention deletion

Let:

```math
a_{it}
=
\exp(s_{it}),
\qquad
A_i
=
\sum_t a_{it}y_{it},
\qquad
S_i
=
\sum_t a_{it}.
```

Then:

```math
z_i
=
\frac{A_i}{S_i}.
```

If token k is removed while all remaining states and scores remain fixed:

```math
z_{i,-k}
=
\frac{
A_i-a_{ik}y_{ik}
}{
S_i-a_{ik}
}.
```

The exact difference is:

```math
z_i-z_{i,-k}
=
\frac{\alpha_{ik}}{1-\alpha_{ik}}
\left(
y_{ik}-z_{i,-k}
\right).
```

This is a readout identity, not a full SSM intervention. Removing the input
patch and rerunning the scan changes every later state:

```math
\Delta_{ik}^{\mathrm{full}}
=
q
\left(
\mathcal C_{\mathrm{SSM}}(U)
\right)
-
q
\left(
\mathcal C_{\mathrm{SSM}}(U\setminus u_{ik})
\right).
```

The full deletion includes the state-memory path, the score path, and the
change in sequence length.

## 7. Strict final-state readout

A final-state readout uses:

```math
z_i^{\mathrm{final}}
=
s_{i,L_i}.
```

For a linear SSM:

```math
z_i^{\mathrm{final}}
=
\sum_{k=1}^{L_i}
\overline A^{L_i-k}
\overline B u_{ik}.
```

The contribution of token k is position-dependent:

```math
c_{ik}
=
\overline A^{L_i-k}\overline B u_{ik}.
```

The first and last tokens have:

```math
c_{i1}
=
\overline A^{L_i-1}\overline B u_{i1},
\qquad
c_{iL_i}
=
\overline B u_{iL_i}.
```

Unless A-bar is identity, the same patch moved to another position changes its
memory multiplier. Final-state readout is therefore a weighted causal summary,
not a normalized first moment.

## 8. Final state and length dependence

Append a suffix V to a prefix U. For a linear recurrence:

```math
s_{L+m}
=
\overline A^m s_L
+
\sum_{r=1}^{m}
\overline A^{m-r}\overline B v_r.
```

The original prefix is multiplied by A-bar to the power m. A suffix can attenuate
or rotate the prefix contribution even when the prefix patches are unchanged.

A final-state model is length-invariant only under restrictive conditions, such
as a specially normalized state or a task head that removes the length effect.
The generic identity is:

```math
s_{L+m}
\ne
s_L
```

when m is positive.

This is a distinct failure from mean length normalization. Mean divides by the
number of outputs; final state does not.

## 9. Max versus mean counterexample

Let two contextualized trajectories be:

```math
Y^{(1)}
=
\begin{bmatrix}
1\\
0\\
0
\end{bmatrix},
\qquad
Y^{(2)}
=
\begin{bmatrix}
1\\
1\\
1
\end{bmatrix}.
```

Their maxima are:

```math
z_{\mathrm{max}}^{(1)}
=
z_{\mathrm{max}}^{(2)}
=
1.
```

Their means are:

```math
z_{\mathrm{mean}}^{(1)}
=
\frac{1}{3},
\qquad
z_{\mathrm{mean}}^{(2)}
=
1.
```

Max cannot distinguish one witness from three equal responses. Mean cannot
distinguish two different sequences with the same average, for example:

```math
Y^{(3)}
=
\begin{bmatrix}
0\\
1\\
2
\end{bmatrix},
\qquad
Y^{(4)}
=
\begin{bmatrix}
1\\
1\\
1
\end{bmatrix},
```

because both means equal one.

## 10. Max versus attention counterexample

Let:

```math
Y
=
\begin{bmatrix}
10\\
0
\end{bmatrix},
\qquad
\alpha
=
\begin{bmatrix}
0.01\\
0.99
\end{bmatrix}.
```

Then:

```math
z^{\mathrm{max}}=10,
\qquad
z^{\mathrm{attn}}=0.1.
```

A large extreme response can be suppressed by attention if its score is low.
Conversely, attention can select a moderate state because its score network
finds it task-aligned.

The two readouts encode different inductive beliefs:

```text
max:
    one coordinatewise extreme is sufficient

attention:
    a learned weighted mixture is sufficient
```

Neither is a universal evidence estimator.

## 11. Attention duplication counterexample

Duplicate token j after the scan with the same state and score. Let:

```math
S
=
\sum_t\exp(s_t).
```

The new denominator is:

```math
S'
=
S+\exp(s_j).
```

The two copies have combined weight:

```math
\alpha_j'+\alpha_{j'}'
=
\frac{2\exp(s_j)}{S+\exp(s_j)}.
```

The representation becomes:

```math
z'
=
\frac{
\sum_{t\ne j}\exp(s_t)y_t
+
2\exp(s_j)y_j
}{
S+\exp(s_j)
}.
```

This generally differs from z. Attention is not multiplicity-invariant, even
when the duplicated states are identical. If duplication happens before C,
the effect is larger because all later states can change.

## 12. Order perturbation and readout invariance

Let P be a permutation applied before the scan. Define:

```math
\Delta_{\mathrm{order}}(P)
=
d
\left(
\mathcal R
\left(
\mathcal C_{\mathrm{SSM}}(PU)
\right),
\mathcal R
\left(
\mathcal C_{\mathrm{SSM}}(U)
\right)
\right).
```

A readout can be row-permutation invariant while the composed model is not:

```math
\mathcal R(PY)=\mathcal R(Y)
\qquad
\not\Longrightarrow
\qquad
\mathcal R(\mathcal C(PU))
=
\mathcal R(\mathcal C(U)).
```

This is the central symmetry distinction for state-space MIL.

A repeated-order ensemble computes:

```math
\overline z
=
\frac{1}{M}
\sum_{m=1}^{M}
\mathcal R
\left(
\mathcal C(P_{\sigma_m}U)
\right).
```

This reduces variance over selected orders but does not make the underlying
single-order model invariant. It also changes compute and can average
biologically incompatible geometries.

## 13. SR-Mamba two-order fusion

For a padded length RN, reshape:

```math
\mathsf X[r,n,:]
=
X_{\mathrm{pad}}[nR+r,:].
```

The original branch scans the flattened order nR+r. The reordered branch scans
rN+n and restores rows with psi:

```math
Y^{\mathrm{orig}}
=
\mathcal C
\left(
X^{\mathrm{orig}}
\right),
```

```math
Y^{\mathrm{reord,restored}}
=
\psi
\left(
\mathcal C
\left(
X^{\mathrm{reord}}
\right)
\right).
```

A residual fusion is schematically:

```math
Y^{\mathrm{SR}}
=
\mathrm{Linear}
\left(
Y^{\mathrm{orig}}
+
Y^{\mathrm{reord,restored}}
\right)
+
X.
```

The final R acts on Y-SR. The two branch outputs can cancel or reinforce:

```math
Y^{\mathrm{orig}}_t
\approx
-
Y^{\mathrm{reord,restored}}_t
\Longrightarrow
Y_t^{\mathrm{SR}}
\approx
\mathrm{Linear}(0)+X_t.
```

A fused readout should therefore be compared with each branch ablated. The
second order is a context change, not a new pooling operator.

## 14. Padding and masks

Let valid-token mask m-it be:

```math
m_{it}
=
\mathbf 1\{t\le L_i\}.
```

A masked mean is:

```math
z_i^{\mathrm{mean}}
=
\frac{
\sum_t m_{it}y_{it}
}{
\sum_t m_{it}
}.
```

A masked attention readout is:

```math
\alpha_{it}
=
\frac{
m_{it}\exp(s_{it})
}{
\sum_u m_{iu}\exp(s_{iu})
}.
```

A masked state update is:

```math
s_{it}
=
m_{it}
\left(
\overline A_t s_{i,t-1}
+
\overline B_tu_{it}
\right)
+
(1-m_{it})s_{i,t-1}.
```

If zero padding is passed through convolution, normalization, or selective
parameter generation, the padded rows can alter valid outputs before they are
removed. Correct de-padding at the end does not repair state contamination.

The padding convention is therefore part of G and C, not only a batching detail.

## 15. Patch-level auxiliary supervision

Suppose token outputs have optional labels q-it and masks r-it. A multitask
objective is:

```math
\mathcal L_i
=
\mathcal L_{\mathrm{slide}}
\left(
y_i,\widehat y_i
\right)
+
\lambda
\sum_t r_{it}
\ell_{\mathrm{patch}}
\left(
q_{it},\widehat q_{it}
\right).
```

The auxiliary term changes the optimization geometry:

```math
\nabla_\theta\mathcal L_i
=
\nabla_\theta\mathcal L_{\mathrm{slide}}
+
\lambda
\nabla_\theta\mathcal L_{\mathrm{patch}}.
```

It can discourage a purely slide-level shortcut, but it also asks the same
contextualized state to support two targets. A patch state that is optimal for
local annotation need not be optimal for the slide statistic.

For S4MIL, max pooling means a patch auxiliary gradient can shape which token
becomes a channelwise maximum. For attention pooling, it can also change score
competition across all tokens.

## 16. Task-head dependence

The same readout vector can feed different heads:

```math
\eta_i^{\mathrm{Cox}}
=
w_{\mathrm{Cox}}^{\mathsf T}z_i,
```

```math
\widehat p_i
=
\mathrm{softmax}
\left(
W_{\mathrm{cls}}z_i+b_{\mathrm{cls}}
\right),
```

```math
\widehat h_i^{(k)}
=
\sigma
\left(
w_k^{\mathsf T}z_i+b_k
\right).
```

A trajectory feature can be useful for classification but irrelevant for a
survival horizon, or useful for risk ordering but poor for calibration. The
appropriate explanation target is:

```math
\Delta_{ik}^{(q)}
=
q_i(U_i)
-
q_i(U_i\setminus u_{ik}),
```

where q is named explicitly.

## 17. Survival readout and time summaries

For discrete hazards:

```math
\widehat h_i^{(k)}
=
\sigma(\ell_i^{(k)}),
\qquad
\widehat S_i(k)
=
\prod_{r=1}^{k}
\left(
1-\widehat h_i^{(r)}
\right).
```

If z is produced by one global sequence readout, all horizons share the same
slide statistic and differ only through task-head parameters. A horizon-specific
readout would instead be:

```math
z_i^{(k)}
=
\mathcal R_{\theta,k}(Y_i),
\qquad
\widehat h_i^{(k)}
=
\sigma
\left(
w_k^{\mathsf T}z_i^{(k)}+b_k
\right).
```

The latter can model time-dependent evidence but increases parameter count,
attention instability, and multiple-horizon interpretation burden.

## 18. Influence paths through the scan

For an output score q, the total derivative with respect to patch u-ik is:

```math
\frac{\partial q}{\partial u_{ik}}
=
\sum_{t=1}^{L_i}
\frac{\partial q}{\partial y_{it}}
\frac{\partial y_{it}}{\partial u_{ik}}
+
\frac{\partial q}{\partial \alpha_i}
\frac{\partial \alpha_i}{\partial u_{ik}}
```

for attention readout, with the second term understood through the score
network. For a causal scan:

```math
\frac{\partial y_{it}}{\partial u_{ik}}=0
\qquad
\text{when }k>t.
```

The gradient support is triangular in sequence coordinates. In a bidirectional
or fused model, the support is broader but remains determined by the chosen
orders and branch wiring.

A heatmap over alpha-it therefore displays only one factor in a chain rule.

## 19. Effective support and entropy

For attention readout, define entropy:

```math
H(\alpha_i)
=
-\sum_t\alpha_{it}\log\alpha_{it}.
```

Define effective support:

```math
N_{\mathrm{eff}}(\alpha_i)
=
\exp(H(\alpha_i)).
```

The same readout dimension can be supported by one nearly dominant token or by
many tokens. A high maximum attention weight does not fully describe support
concentration.

For max, define the tie set:

```math
T_{ir}
=
\left\{
t:[y_{it}]_r=[z_i^{\mathrm{max}}]_r
\right\}.
```

A large tie set means the max statistic is insensitive to perturbations that
leave all maxima unchanged. A singleton max creates a sharper but potentially
unstable witness.

## 20. Sequence length and sampling density

Let a slide be sampled at two densities with lengths L and L-prime. If the
sequence is a replicated or refined representation, mean after a fixed
context may be approximately stable, but max and attention need not be:

```math
z^{\mathrm{mean}}(Y')
\approx
z^{\mathrm{mean}}(Y)
\qquad
\not\Longrightarrow
\qquad
z^{\mathrm{max}}(Y')
=
z^{\mathrm{max}}(Y).
```

Adding one high-response token changes max immediately:

```math
[z^{\mathrm{max}}(Y\cup y_{\mathrm{new}})]_r
=
\max
\left(
[z^{\mathrm{max}}(Y)]_r,
[y_{\mathrm{new}}]_r
\right).
```

Attention changes all normalized masses when a token is added. A deployment
comparison should match patch count, tissue area, and valid-token mask before
attributing a shift to biology.

## 21. Readout collision matrix

| Readout | Invariant after scan? | Preserves | Loses |
| --- | --- | --- | --- |
| max | output-row permutations and duplicated fixed output rows | coordinatewise extremes | multiplicity, order, submaximal values |
| mean | output-row permutations and duplicated fixed output rows | first moment | higher moments and positions |
| attention | output-row permutations with matched scores | learned weighted first moment | alternative score/value mixtures |
| final state | no generic row permutation invariance | position-weighted finite-state summary | all state-nullspace prefixes |
| SR fusion readout | only the selected branch symmetries | information shared by both trajectories and residual | branch-specific cancellations |

The phrase “the SSM aggregates the bag” is incomplete without this matrix.

## 22. C/R/G/S design matrix

| Method | C: context | R: readout | G: order geometry | S: supervision |
| --- | --- | --- | --- | --- |
| S4MIL | structured S4D convolution over ordered patch features | coordinatewise max | inherited extraction or grid order | slide label plus optional patch auxiliary loss |
| MambaMIL | selective causal Mamba blocks with convolution, gates, and residuals | released-code attention weighted mean | original sequence order | classification or discrete hazard head |
| BiMamba | forward and reverse selective branches | task-specific sequence readout | two directions of chosen order | slide-level task loss |
| SR-Mamba | original and segment-transposed selective branches | fused trajectory readout | original plus segment-transposed order | slide-level task loss |

The table separates the paper-level context operator from the readout. A
survival head is S, not an intrinsic consequence of the SSM.

## 23. Sanity-check suite

### 23.1 Order permutation

```math
\Delta_{\sigma,\sigma'}
=
d
\left(
F(P_\sigma U),
F(P_{\sigma'}U)
\right).
```

Report multiple order families and the distribution of the result.

### 23.2 Readout isolation

Hold Y fixed and compare:

```math
z^{\mathrm{max}},
\qquad
z^{\mathrm{mean}},
\qquad
z^{\mathrm{attn}},
\qquad
z^{\mathrm{final}}.
```

This separates R from C.

### 23.3 Scan ablation

Compare:

```math
F_{\mathrm{SSM}},
\qquad
F_{\mathrm{identity}},
\qquad
F_{\mathrm{randomized\ order}}.
```

The identity branch measures how much of the result comes from the residual or
tokenwise path.

### 23.4 Direction and SR ablation

For SR-Mamba compare original, reordered, and fused branches. For bidirectional
models compare forward, reverse, and combined outputs.

### 23.5 Padding invariance

Append masked padding and verify:

```math
F(U)
=
F(U\Vert 0_{\mathrm{pad}})
```

under the stated mask convention.

### 23.6 Deletion recomputation

Remove a patch and recompute all later states, scores, masks, and readout
normalization. Report this separately from fixed-trajectory attention credit.

## 24. Reporting rule

A state-space MIL method is minimally specified when it reports:

```text
1. patch feature dimension and token count
2. exact sequence ordering rule
3. scan direction and whether multiple branches exist
4. continuous or selective state parameterization
5. state width, channel width, and block count
6. convolution, gate, normalization, and residual paths
7. padding and mask behavior
8. exact readout and valid-token support
9. task head and loss
10. order, length, density, direction, and readout ablations
```

The compact decomposition is:

```math
\begin{aligned}
G &: \text{ordered patch geometry and branch permutations},
\\
C &: \text{finite-state recurrent or convolutional context},
\\
R &: \text{max, mean, attention, final state, or branch fusion},
\\
S &: \text{slide, patch, survival, and auxiliary supervision}.
\end{aligned}
```

## 25. Bottom line

State-space MIL is:

```math
\boxed{
\text{patch multiset}
\longrightarrow
\text{ordered sequence}
\longrightarrow
\text{finite selective trajectory}
\longrightarrow
\text{surviving statistic}
\longrightarrow
\text{task output}
}
```

The state-space scan can be linear in sequence length without being
permutation-invariant, lossless, or order-free. The readout determines whether
the model retains a witness, a first moment, an attention-weighted mixture, a
position-weighted final state, or a fused multi-order trajectory.

The correct question is:

```math
\boxed{
\text{what information about the ordered contextual trajectory remains after R,
and which supervision S makes that information useful?}
}
```
