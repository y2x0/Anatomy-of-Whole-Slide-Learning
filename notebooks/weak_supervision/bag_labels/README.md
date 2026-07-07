# Bag Labels

Bag labels are the default supervision object in WSI MIL.

The observed signal is:

```math
S_i
=
Y_i,
```

while instance labels are hidden:

```math
Z_{ij}
\text{ unobserved}.
```

## Files

- `01_latent_instance_labels_and_bag_map.md`: latent patch labels and bag maps.
- `02_standard_mil_assumptions.md`: max, collective, threshold, and distributional MIL.
- `03_marginal_likelihood_and_posterior_ambiguity.md`: bag likelihood and non-identifiability.
- `04_positive_unlabeled_bag_view.md`: negative bags, positive bags, and PU-style risk.
- `05_failure_modes.md`: witness ambiguity, shortcut witnesses, and prevalence mismatch.

## C/R/G/S Placement

```text
G:
    optional geometry, usually hidden from the label

C:
    instance feature map and optional context

R:
    bag map from latent instances to slide prediction

S:
    slide-level label only
```

Bag labels supervise the bag event. They do not directly supervise the patch
explanation.
