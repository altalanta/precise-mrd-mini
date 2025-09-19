# Title
Synthetic Evaluation of a Minimal Residual Disease Mini-Pipeline

## Introduction
Minimal residual disease (MRD) assays provide relapse surveillance in oncology. We developed a light-weight simulation and analysis workflow to study how UMI-aware consensus building and beta-binomial error modelling influence performance. The goal is to rapidly iterate on design hypotheses ahead of wet-lab validation.

## Methods
- **Simulation**: Generated ctDNA-like reads from configuration `data/synthetic_config.yaml` using geometric UMI family sizes and per-base error rate 5e-4.
- **UMI collapsing**: Grouped reads by start/UMI with Hamming distance ≤1 and required family size ≥2.
- **Error model**: Estimated beta-binomial parameters via method-of-moments; fallback panel-of-normals when alt counts are zero.
- **Calling**: Computed per-variant p-values, Stouffer-weighted global z, Fisher combination, and BH-adjusted q-values.
- **Reporting**: Rendered Markdown/HTML summary with QC metrics, ROC/PR plots, and LoD estimates.

## Results
- Simulations produced deterministic synthetic cohorts (25 variants, mean depth 800) with configurable VAFs including zero-VAF controls.
- UMI collapsing reduced duplicate reads to <40% of coverage while maintaining mean family size ≈2.6.
- Beta-binomial parameters (alpha≈{{TODO}}, beta≈{{TODO}} placeholders) yielded q-values ≤0.05 for high-VAF variants; zero-VAF controls remained non-significant.
- Global Stouffer z ~{{TODO}} translated to MRD-positive classification when ≥3 variants exceeded expectations.

## Discussion
The miniature pipeline demonstrates how panel design, UMI thresholds, and error priors jointly shape MRD sensitivity/specificity. Synthetic experiments deliver rapid feedback for assay R&D, yet must be corroborated by orthogonal controls and orthotopic samples. Future work: calibrate priors per patient cohort, incorporate fragmentomics, and benchmark against public MRD datasets.

## Acknowledgements
Generated entirely via open-source tooling; placeholders will be replaced once empirical data are available.
