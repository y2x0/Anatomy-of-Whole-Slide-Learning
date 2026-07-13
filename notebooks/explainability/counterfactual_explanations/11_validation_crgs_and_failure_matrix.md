# Counterfactual Validation, C/R/G/S, and Failure Matrix

## 1. Validation Axes

No single metric validates a counterfactual. Report:

```math
\begin{aligned}
\text{target success:}&\quad
\mathbf 1\{f(x_{\mathrm{cf}})\in\mathcal Y_{\star}\},\\
\text{proximity:}&\quad d(x,x_{\mathrm{cf}}),\\
\text{sparsity:}&\quad
\|x-x_{\mathrm{cf}}\|_0,\\
\text{plausibility:}&\quad
\text{density, realism, or expert assessment},\\
\text{stability:}&\quad
d(x_{\mathrm{cf}},\widetilde x_{\mathrm{cf}}),\\
\text{diversity:}&\quad
\det K\text{ or mechanism-aware separation}.
\end{aligned}
```

For images, reconstruction quality, FID, SSIM, blinded expert realism, and
quantified cell or tissue changes test different properties.

## 2. C/R/G/S Placement

| Method | Counterfactual state | Context `C` | Readout `R` | Geometry `G` | Supervision `S` |
|---|---|---|---|---|---|
| Wachter | tabular feature vector | fixed black-box score | target penalty plus distance | user-selected feature metric | desired output |
| DiCE | set of feature vectors | joint DPP coupling | valid diverse solution set | proximity and kernel geometry | desired output plus constraints |
| HIPPO | WSI patch multiset | full model recomputation | deletion or addition effect | bag membership and optional annotations | model output or tissue hypothesis |
| GMM-CeFlow | interpretable WSI feature vector | invertible flow and Gaussian classes | nearest quadratic-boundary projection | learned latent density | class labels and HIFs |
| MoPaDi-linear | patch image through latent feature | linear classifier direction and DDIM | generated image trajectory | diffusion latent and image geometry | patch class labels |
| MoPaDi-MIL | selected patch within WSI bag | bag-conditional target gradient and DDIM | generated selected tiles | contextual MIL and diffusion geometry | slide labels |

## 3. Failure Matrix

| Failure | Cause | Required check |
|---|---|---|
| invalid target | optimization or decoding misses class | evaluate final generated object |
| metric artifact | closeness metric favors meaningless edit | compare pathology-aware costs |
| infeasible combination | independent feature changes violate biology | constraints or structural model |
| surrogate mismatch | explanation flips `g` but not original `F` | local fidelity near boundary |
| flow inversion distortion | sparse latent edit becomes dense HIF edit | report both latent and feature changes |
| generator hallucination | decoder creates unsupported morphology | blinded experts and quantitative morphometry |
| gradient overshoot | local direction used too far from origin | score path and curvature diagnostics |
| context mismatch | tiles edited separately but bag interactions change | reassemble and rerun full bag |
| nonunique explanation | many equal-cost boundary crossings exist | diverse solution set and stability |
| causal overclaim | model response labeled biological mechanism | state intervention semantics explicitly |

## 4. Reporting Rule

Every WSI counterfactual should state:

```text
edited state space; target model; desired output; distance; feasibility set;
optimization or generator; selected patches; context recomputation; target
success; semantic plausibility; diversity; surrogate fidelity; causal scope.
```

The safest conclusion is model-relative: under the specified edit mechanism,
the trained system changes its output. Biological interpretation requires
independent validation.

