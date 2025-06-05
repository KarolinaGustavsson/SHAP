### shap_analysis.py
import os
os.environ["OMP_NUM_THREADS"] = "1"  # 🔐 limit threading to avoid crash

# 🔧 Added for macOS stability
import matplotlib
matplotlib.use('Agg')  # ⮆️ prevent crashes from GUI rendering

import multiprocessing as mp
mp.set_start_method("spawn", force=True)  # ⮆️ safer than fork() on macOS

from model import AgeRegressor
from utils import load_data
from shap_utils import explain_model_with_shap
import torch

def main():
    print("🔍 Running SHAP analysis...")
    df_norm, _, _, _, ages = load_data("../data/scrambled.csv")

    all_features = df_norm.drop("age", axis=1)
    x = torch.tensor(all_features.values, dtype=torch.float32)
    feature_names = [col for col in all_features.columns if col != "age"]

    # Load trained model
    model = AgeRegressor(input_dim=x.shape[1])
    checkpoint = torch.load("amoris_model.ckpt", map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["state_dict"])
    model.model.to(torch.device("cpu"))

    explain_model_with_shap(model.model, x, ages, feature_names, outdir="../results")

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
