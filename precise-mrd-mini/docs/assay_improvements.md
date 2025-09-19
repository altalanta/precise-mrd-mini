# Assay Improvements Notes

- **Panel size**: Expanding the variant panel improves aggregate sensitivity via Stouffer weighting while diluting per-site error; cite `[Ref: pending]`.
- **UMI minimum family size**: Raising the threshold suppresses sequencing noise but can increase drop-out at low input; choose min_size=2 as compromise.
- **Error model stringency**: Tight beta-binomial priors (larger alpha+beta) reduce false positives but risk under-calling emerging clones; plan adaptive priors per sequencing batch `[Ref: pending]`.
