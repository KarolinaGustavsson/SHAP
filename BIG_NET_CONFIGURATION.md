# Big Net Configuration Guide

The big net (RandomForest) can now be trained on different targets. Easily switch between approaches by modifying the configuration in `py/shap_analysis.py`.

## Configuration Options

Located at the top of `py/shap_analysis.py`:

```python
# ============================================================================
# CONFIGURATION: Switch between different big net modes
# ============================================================================
BIG_NET_MODE = "residuals"  # Options: "residuals", "kdm_advance", "phenoage_advance"
INCLUDE_STATUS = False  # Only used with phenoage_advance
# ============================================================================
```

## Three Modes

### 1. **"residuals"** (Original / Default)
- **What it does**: Big net learns to predict age gaps (y_pred - y_true)
- **Questions answered**: "Which biomarkers cause the small net's prediction errors?"
- **Status**: Not used
- **Configuration**:
  ```python
  BIG_NET_MODE = "residuals"
  INCLUDE_STATUS = False  # Ignored in this mode
  ```

### 2. **"kdm_advance"** (KDM Biological Age)
- **What it does**: Big net learns to predict kdm_advance values directly
- **Questions answered**: "Which biomarkers drive KDM biological age?"
- **Status**: Not used
- **Configuration**:
  ```python
  BIG_NET_MODE = "kdm_advance"
  INCLUDE_STATUS = False  # Status not relevant for KDM
  ```

### 3. **"phenoage_advance"** (PhenoAge Biological Age)
- **What it does**: Big net learns to predict phenoage_advance values, optionally with mortality status
- **Questions answered**: "Which biomarkers drive PhenoAge biological age?" and optionally "How does mortality status interact with biomarkers?"
- **Status**: Can be optionally included as input feature
- **Configuration options**:
  
  **Without mortality status**:
  ```python
  BIG_NET_MODE = "phenoage_advance"
  INCLUDE_STATUS = False
  ```
  
  **With mortality status** (more complex model):
  ```python
  BIG_NET_MODE = "phenoage_advance"
  INCLUDE_STATUS = True  # Adds status as input feature
  ```

## Example Usage

To switch modes, edit `py/shap_analysis.py` and change the configuration at the top:

### Example 1: Compare residuals vs biological age
```python
# Run with residuals approach
BIG_NET_MODE = "residuals"
# Run shap_analysis.py

# Then switch to:
BIG_NET_MODE = "phenoage_advance"
INCLUDE_STATUS = False
# Run shap_analysis.py again

# Compare results in ../results/
```

### Example 2: Explore mortality interactions
```python
BIG_NET_MODE = "phenoage_advance"
INCLUDE_STATUS = True
# Run shap_analysis.py
# Check SHAP plots to see status importance
```

## Technical Details

### Data Loading
- **"residuals" mode**: Uses `load_data()` - loads 18 biomarkers + age + status
- **"kdm_advance"/"phenoage_advance" modes**: Uses `load_data_with_bioage()` - additionally loads kdm_advance and phenoage_advance

### Feature List
- **All modes except phenoage with status**: Uses 18 biomarkers
- **phenoage_advance with INCLUDE_STATUS=True**: Uses 18 biomarkers + status (19 features total)

### SHAP Output
- **Residual SHAP summary plot**: `shap_bioage_summary.png` (renamed from `shap_residual_summary.png`)
- **Feature importance**: Title dynamically changes based on target (e.g., "Age Gap (Residuals)" vs "PhenoAge Advance")

## Reverting to Original Approach

If you want to go back to the original residual-based approach:
```python
BIG_NET_MODE = "residuals"
INCLUDE_STATUS = False
```

This is the default configuration and matches the original code behavior.
