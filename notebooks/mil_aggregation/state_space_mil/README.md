# State-Space MIL

This portion asks:

`text
How does an ordered patch bag become a slide statistic through a finite state?
`

State-space MIL changes the slide object before aggregation:

`math
\mathcal{B}_i
\xrightarrow{\;\sigma_i\;}
U_i
\xrightarrow{\;\mathcal{C}_{\mathrm{SSM}}\;}
Y_i
\xrightarrow{\;\mathcal{R}\;}
z_i
\xrightarrow{\;\mathcal{H}\;}
\widehat y_i.
`

The order is part of the representation, not an implementation detail. A
state-space operator may have linear sequence-length complexity while remaining
order-sensitive and lossy through its finite state dimension.

## Notes

`text
01_ordered_bag_and_state_space_symmetry.md
    sequence object, permutation dependence, padding, and shape map

02_linear_state_space_discretization_and_convolution.md
    continuous SSMs, zero-order hold, recurrent form, and S4 convolution

03_s4mil_structured_state_space_readout.md
    mapped 2023 pathology SSM pipeline, max readout, and multitask loss

04_selective_state_spaces_and_mamba_scan.md
    input-dependent selection, discretization, causal recurrence, and cost

05_mambamil_vanilla_scan_and_mil_head.md
    paper and released-code MambaMIL forward maps

06_srmamba_sequence_reordering_operator.md
    exact segment reshape, transposition, restoration, gating, and residual

07_state_space_readout_and_surviving_statistics.md
    max, mean, attention, final-state readouts, and task supervision

08_state_space_mil_failure_modes_and_crgs.md
    order artifacts, state compression, padding, long-range loss, and synthesis
`

## Source Boundary

The deep derivations use mapped sources already present in the private research
map:

- Fillioux et al., *Structured State Space Models for Multiple Instance
  Learning in Digital Pathology* / S4MIL. MICCAI 2023.
  https://arxiv.org/abs/2306.15789
- Yang, Wang, and Chen, *MambaMIL: Enhancing Long Sequence Modeling with
  Sequence Reordering in Computational Pathology*. MICCAI 2024.
  https://arxiv.org/abs/2403.06800
- Gu, Goel, and Re, *Efficiently Modeling Long Sequences with Structured State
  Spaces* / S4. https://arxiv.org/abs/2111.00396
- Gu and Dao, *Mamba: Linear-Time Sequence Modeling with Selective State
  Spaces*. https://arxiv.org/abs/2312.00752

Later 2025 WSI extensions remain separate until their forward maps are derived
directly.
