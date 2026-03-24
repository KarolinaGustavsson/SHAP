### shap_analysis.py
import os
os.environ["OMP_NUM_THREADS"] = "1"  # 🔐 limit threading to avoid crash

# 🔧 Added for macOS stability
import matplotlib
matplotlib.use('Agg')  # ⮆️ prevent crashes from GUI rendering

import multiprocessing as mp
mp.set_start_method("spawn", force=True)  # ⮆️ safer than fork() on macOS

from model import AgeRegressor
from utils import load_data, load_data_with_bioage
from shap_utils import explain_model_with_shap
import torch

# ============================================================================
# CONFIGURATION: Switch between different big net modes
# ============================================================================
BIG_NET_MODE = "residuals"  # Options: "residuals", "kdm_advance", "phenoage_advance"
INCLUDE_STATUS = False  # Only used with phenoage_advance
INCLUDE_AGE_IN_BIGNET = False # Only used with phenoage_advance: include chronological age as input to big net
# ============================================================================

# NOTE ON ARCHITECTURE AND STATUS/MORTALITY:
# Mortality status and follow-up time are CONSEQUENCES of aging/health, not DRIVERS of biological
# aging itself. Therefore, they should NOT be used as input features in the big net, even though
# they are correlated with biological age. Including them would introduce data leakage and survival
# bias: alive people with long follow-up times would be treated differently than dead people, leading
# the model to learn spurious survival patterns rather than true biological aging patterns.
# 
# Chronological age (INCLUDE_AGE_IN_BIGNET), however, IS appropriate as input for phenoage_advance
# because we want to understand how biomarkers + age together predict the deviation from expected
# biological age (phenoage_advance = biological age - chronological age effects).
#
# If you wanted to model mortality properly, you would use survival analysis (Cox proportional
# hazards model) with time+status as the outcome (not inputs), similar to Morgan-Levine's approach.
# ============================================================================

def main():
    print("🔍 Running SHAP analysis...")
    
    # Load data based on big net mode
    if BIG_NET_MODE == "residuals":
        # Original approach: use only age and biomarkers
        df_norm, _, _, _, ages = load_data("../data/mimic_b.csv") #mimic_b as default
        y_bioage = None
        status = None
    else:
        # New approach: load biological age measures
        df_norm, _, _, status, ages, kdm_advance, phenoage_advance = load_data_with_bioage("../data/mimic_b.csv")
        if BIG_NET_MODE == "kdm_advance":
            y_bioage = kdm_advance
        elif BIG_NET_MODE == "phenoage_advance":
            y_bioage = phenoage_advance
        else:
            raise ValueError(f"Unknown BIG_NET_MODE: {BIG_NET_MODE}")
    
    x = torch.tensor(df_norm.drop("age", axis=1).values, dtype=torch.float32)
    feature_names = df_norm.drop("age", axis=1).columns

    # Load trained model
    model = AgeRegressor(input_dim=x.shape[1])
    checkpoint = torch.load("amoris_model.ckpt", map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["state_dict"])
    model.model.to(torch.device("cpu"))

    print(f"📊 Using BIG_NET_MODE: {BIG_NET_MODE}")
    if BIG_NET_MODE == "phenoage_advance" and INCLUDE_AGE_IN_BIGNET:
        print("✅ Including chronological age as input feature for big net")
    
    explain_model_with_shap(
        model.model, x, ages, feature_names, 
        outdir="../results",
        big_net_mode=BIG_NET_MODE,
        y_bioage=y_bioage,
        status=status,
        include_status=INCLUDE_STATUS,
        include_age_in_bignet=INCLUDE_AGE_IN_BIGNET,
        chronological_ages=ages
    )

if __name__ == "__main__":
    main()

##from model import AgeRegressor
##from utils import load_data
##from shap_utils import explain_model_with_shap
##import torch
##
##def main():
##    df_norm, _, _, _, ages = load_data("../data/scrambled.csv")
##    x = torch.tensor(df_norm.drop("age", axis=1).values, dtype=torch.float32)
##    feature_names = df_norm.drop("age", axis=1).columns
##
##    # Load trained model
##    model = AgeRegressor(input_dim=x.shape[1])
##    checkpoint = torch.load("amoris_model.ckpt", map_location=torch.device("cpu"))
##    model.load_state_dict(checkpoint["state_dict"])
##
##    explain_model_with_shap(model.model, x, ages, feature_names, outdir="../results")
##
##if __name__ == "__main__":
##    main()
