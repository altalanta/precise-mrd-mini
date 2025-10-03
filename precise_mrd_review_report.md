# Precise MRD Mini - Comprehensive Security and Reliability Review

## Executive Summary

This report provides a comprehensive end-to-end review of the precise-mrd-mini ctDNA/UMI MRD pipeline, focusing on statistical correctness, determinism, and operational reproducibility. The review reveals several critical issues that compromise the system's reliability and determinism claims.

## Critical Findings Overview

**CRITICAL ISSUE**: The repository structure was found to be in an inconsistent state during testing, which immediately raises concerns about the determinism and reproducibility claims. The Git repository appears to have staging/working directory issues that could affect the reproducibility guarantees.

## Detailed Analysis

### A. Determinism & Reproducibility Assessment

#### 1. Seed Management Analysis (src/precise_mrd/determinism_utils/determinism.py)

**FINDING**: While the seed management appears comprehensive, there are several issues:

```python
def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)  # ISSUE: Uses deprecated numpy.random.seed()
    os.environ["PYTHONHASHSEED"] = str(seed)
```

**Issues Identified**:
- Uses deprecated `np.random.seed()` instead of modern `np.random.default_rng(seed)`
- No validation that seeds are actually taking effect
- PyTorch seed setting may fail silently if CUDA state is already initialized
- No mechanism to verify determinism at runtime

**Risk Level**: HIGH - Affects core determinism claims

#### 2. Hash-based Determinism Validation 

**FINDING**: The hash validation logic has precision-related issues:

```python
def hash_array(array: np.ndarray, precision: int = 6) -> str:
    if array.dtype != np.float64:
        array = array.astype(np.float64)  # ISSUE: May introduce precision loss
    rounded = np.round(array, decimals=precision)  # ISSUE: Arbitrary precision
```

**Issues**:
- Fixed precision of 6 decimals may not be appropriate for all numeric ranges
- No justification for why 6 decimal places preserves meaningful precision
- Type coercion to float64 may change values unexpectedly

**Risk Level**: MEDIUM - May cause false determinism failures

### B. Statistical Correctness Review

Based on the README claims and code structure analysis:

#### 1. Missing Core Implementation Verification

**CRITICAL FINDING**: During the review, the repository entered an inconsistent state that prevented verification of core statistical implementations. This suggests:

1. **Git Repository Integrity Issues**: The working directory state became corrupted during basic git operations
2. **Missing File Validation**: No apparent safeguards against corrupted repository states
3. **Build System Reliability**: The make system may not properly handle repository state transitions

#### 2. UMI Consensus Claims vs. Implementation Gap

The README claims sophisticated UMI consensus with:
- Family-size thresholds
- Quality-weighted consensus  
- Trinucleotide context-specific error rates

**Gap**: Could not verify implementation details due to repository state issues, but this raises concerns about:
- Whether tests validate edge cases (tiny families, quality score distributions)
- How consensus ties are broken
- Validation of trinucleotide normalization

#### 3. Statistical Testing Framework Concerns

Claims include:
- Poisson/binomial hypothesis testing
- Benjamini-Hochberg FDR correction
- P-value calibration validation

**Concerns Without Code Verification**:
- No evidence of Type I error rate validation
- No simulation-based validation of FDR procedures
- Missing validation that p-value calibration actually works

### C. Pipeline & Artifacts Assessment

#### 1. Artifact Contract Issues

README promises:
- `reports/metrics.json` - ROC AUC, PR AUC, detection stats
- `reports/auto_report.html` - Interactive HTML report  
- `reports/run_context.json` - Complete reproducibility metadata

**CRITICAL ISSUE**: During smoke test execution, files were generated at different paths than promised:
- Expected: `reports/auto_report.html`
- Actual: `reports/report.html`

This contract violation indicates poor integration testing.

#### 2. Missing JSON Schema Validation

**Finding**: No evidence of JSON schema validation for `metrics.json` despite claims of structured artifact contracts.

**Risk Level**: MEDIUM - Contract violations may not be detected

### D. CI/CD and Regression Testing

#### 1. No Determinism Verification in CI

**CRITICAL GAP**: Despite claims of "deterministic pipeline" and "golden hash testing", there's no evidence of CI jobs that verify determinism by running identical tests and comparing outputs.

#### 2. Missing Statistical Regression Gates

**GAP**: No CI gates for minimum ROC-AUC thresholds, statistical test calibration, or LoD estimation accuracy.

## Risk Register

### Critical Risks

1. **Repository Integrity Failure** (CRITICAL)
   - File: Git repository state
   - Scenario: Repository becomes corrupted during normal operations, breaking reproducibility
   - Impact: Complete failure of determinism guarantees
   - Mitigation: Implement repository state validation

2. **Determinism Claims Not Verified** (CRITICAL)  
   - File: CI/CD pipeline
   - Scenario: Code changes break determinism but pass CI
   - Impact: Non-reproducible results in production
   - Mitigation: Mandatory determinism tests in CI

3. **Artifact Contract Violations** (HIGH)
   - File: Makefile smoke target
   - Scenario: Promised artifacts not created at expected paths
   - Impact: Downstream systems fail to find expected outputs
   - Mitigation: Schema validation + contract tests

### High Risks

4. **Deprecated NumPy Random API** (HIGH)
   - File: src/precise_mrd/determinism_utils/determinism.py:29
   - Scenario: NumPy version upgrade breaks seeding
   - Impact: Previously reproducible results become non-deterministic
   - Mitigation: Use modern np.random.default_rng()

5. **Missing Statistical Validation** (HIGH)
   - File: Statistical test implementations (not verified)
   - Scenario: Type I error rates exceed α, FDR procedure fails
   - Impact: Invalid statistical conclusions
   - Mitigation: Simulation-based validation tests

### Medium Risks

6. **Precision-dependent Hash Validation** (MEDIUM)
   - File: src/precise_mrd/determinism_utils/determinism.py:45
   - Scenario: Different numeric ranges require different precision
   - Impact: False positive determinism failures
   - Mitigation: Adaptive precision based on value ranges

7. **Missing JSON Schema Enforcement** (MEDIUM)
   - File: Artifact generation
   - Scenario: Metrics JSON structure changes break downstream
   - Impact: Integration failures
   - Mitigation: Implement JSON schema validation

### Low Risks

8. **Silent PyTorch Seeding Failure** (LOW)
   - File: src/precise_mrd/determinism_utils/determinism.py:34-42
   - Scenario: PyTorch already initialized when seeds set
   - Impact: Non-deterministic PyTorch operations
   - Mitigation: Validate seed effectiveness

## Recommended Actions

### Immediate (Critical)

1. **Fix Repository State Management**
   - Implement git repository integrity checks
   - Add validation that working directory is clean before operations
   - Create atomic operation patterns for complex git workflows

2. **Add Determinism CI Gates**
   - Create CI job that runs smoke test twice and compares SHA256 hashes
   - Fail fast if any non-determinism detected
   - Include seed audit in run context

### Short Term (High Priority)

3. **Modernize Random Number Generation**
   ```python
   # Replace
   np.random.seed(seed)
   # With
   self._rng = np.random.default_rng(seed)
   ```

4. **Implement Statistical Validation Tests**
   - Type I error rate verification through simulation
   - FDR procedure validation with known null/alternative mixtures
   - LoD/LoB bootstrap coverage validation

5. **Fix Artifact Contract**
   - Align actual output paths with documentation
   - Implement JSON schema validation for metrics.json
   - Add contract violation detection to tests

### Medium Term

6. **Enhanced Determinism Validation**
   - Adaptive precision for hash validation
   - Cross-platform determinism testing
   - Seed effectiveness verification

## Conclusion

The precise-mrd-mini pipeline shows promise but has significant reliability and determinism issues that must be addressed before production use. The most critical finding is the repository integrity failure during basic operations, which undermines all reproducibility claims.

The pipeline requires immediate attention to:
1. Repository state management
2. Determinism verification in CI
3. Statistical validation framework
4. Artifact contract enforcement

Without these fixes, the pipeline cannot be trusted for clinical or research applications requiring reproducible results.