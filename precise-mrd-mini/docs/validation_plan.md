# Analytical Validation Outline (Synthetic)

1. **Limit of detection (LoD)**
   - Use the synthetic sweeps (reports/figures/lod_grid.csv) to confirm LoD50/LoD95 across depth and VAF combinations.
2. **Precision / Reproducibility**
   - Re-run the Snakefile with varied seeds; compare MRD calls and QC metrics via coefficient of variation.
3. **Linearity**
   - Titrate VAFs (0.01%–1%) and fit linear regression of observed vs expected alt fractions (statsmodels).
4. **Interference**
   - Spike-in negative control variants (vaf_true=0) to ensure beta-binomial FDR stays <5%.
5. **Carryover / Cross-contamination**
   - Alternate high/low VAF simulations; quantify cross-sample UMIs.
6. **Stability**
   - Simulate storage by reprocessing generated parquet files after random perturbations; confirm no drift beyond QC thresholds.

_All statements limited to synthetic data pending wet-lab confirmation._
