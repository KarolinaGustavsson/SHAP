### shap_utils.py
import os
os.environ["OMP_NUM_THREADS"] = "1"  # 🔐 limit threading to avoid crash

import torch
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

class WrappedModel(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model

    def forward(self, x):
        if isinstance(x, np.ndarray):
            x = torch.tensor(x, dtype=torch.float32)
        return self.base(x)

def plot_shap_residual_vs_prediction(shap_pred_values, shap_resid_values, feature_names, outpath):
    shap_pred_values = np.array(shap_pred_values)
    if hasattr(shap_resid_values, "values"):
        shap_resid_values = np.array(shap_resid_values.values)
    else:
        shap_resid_values = np.array(shap_resid_values)

    for i, fname in enumerate(feature_names):
        plt.figure(figsize=(5, 5))
        plt.scatter(shap_pred_values[:, i], shap_resid_values[:, i], alpha=0.4)
        plt.xlabel(f"SHAP for Prediction ({fname})")
        plt.ylabel(f"SHAP for Residual ({fname})")
        plt.axhline(0, color='gray', linestyle='--')
        plt.axvline(0, color='gray', linestyle='--')
        plt.title(f"{fname}: Residual SHAP vs. Prediction SHAP")
        plt.grid(True)
        plt.savefig(f"{outpath}/shap_resid_vs_pred_{fname}.png", bbox_inches="tight")
        plt.close()

def explain_residuals(x_df, y_true, y_pred, feature_names, outdir="results", x_explain=None, 
                      big_net_mode="residuals", y_bioage=None, status=None, include_status=False,
                      include_age_in_bignet=False, chronological_ages=None):
    """
    Explain prediction errors using RandomForest (big net).
    
    Parameters:
    -----------
    big_net_mode : str
        "residuals" : Train on y_pred - y_true (default, original behavior)
        "kdm_advance" : Train on kdm_advance values (requires y_bioage)
        "phenoage_advance" : Train on phenoage_advance values (requires y_bioage)
    y_bioage : array-like, optional
        Biological age values (kdm_advance or phenoage_advance)
    status : array-like, optional
        Mortality status (0=alive, 1=dead), used when include_status=True
    include_status : bool
        If True and big_net_mode is phenoage, include status as input feature
    """
    # Determine target variable for big net
    if big_net_mode == "residuals":
        target = y_pred - y_true
        target_name = "Age Gap (Residuals)"
    elif big_net_mode == "kdm_advance":
        if y_bioage is None:
            raise ValueError("y_bioage must be provided for big_net_mode='kdm_advance'")
        target = y_bioage
        target_name = "KDM Advance"
    elif big_net_mode == "phenoage_advance":
        if y_bioage is None:
            raise ValueError("y_bioage must be provided for big_net_mode='phenoage_advance'")
        target = y_bioage
        target_name = "PhenoAge Advance"
    else:
        raise ValueError(f"Unknown big_net_mode: {big_net_mode}")

    # Prepare features for big net
    x_data = x_df.values
    feature_list = list(feature_names)
    
    # Add age if requested (only for phenoage_advance)
    if include_age_in_bignet and chronological_ages is not None and big_net_mode == "phenoage_advance":
        x_data = np.column_stack([x_data, chronological_ages])
        feature_list = list(feature_list) + ["age"]
        print("✅ Including chronological age as input feature for phenoage_advance model")
    
    # Add status if requested
    if include_status and status is not None and big_net_mode == "phenoage_advance":
        x_data = np.column_stack([x_data, status])
        feature_list = list(feature_list) + ["status"]
        print("✅ Including 'status' as input feature for phenoage_advance model")
    elif include_status and status is not None:
        print(f"⚠️ include_status=True but big_net_mode='{big_net_mode}'. Status only added for phenoage_advance.")

    # Train/test split for training the residual model
    x_train, _, y_train, _ = train_test_split(
        x_data, target, test_size=0.2, random_state=42
    )

    # Fit big net model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    
    print(f"✅ Big Net trained on: {target_name}")

    # Use subset for SHAP
    if x_explain is None:
        sample_idx = np.random.choice(len(x_data), size=min(100, len(x_data)), replace=False)
        if include_age_in_bignet and chronological_ages is not None and big_net_mode == "phenoage_advance":
            # Sample includes age
            x_explain_base = x_df.iloc[sample_idx].values
            age_subset = chronological_ages[sample_idx]
            x_explain = np.column_stack([x_explain_base, age_subset])
            if include_status and status is not None:
                status_subset = status[sample_idx]
                x_explain = np.column_stack([x_explain, status_subset])
        elif include_status and status is not None and big_net_mode == "phenoage_advance":
            # Sample includes status only
            x_explain_base = x_df.iloc[sample_idx].values
            status_subset = status[sample_idx]
            x_explain = np.column_stack([x_explain_base, status_subset])
        else:
            x_explain = x_data[sample_idx]
    else:
        # x_explain was passed in - need to augment it if include_age_in_bignet
        if include_age_in_bignet and chronological_ages is not None and big_net_mode == "phenoage_advance":
            # x_explain has only biomarkers, need to add age
            # Find which indices from the original data correspond to this x_explain subset
            sample_idx = np.arange(len(x_explain))
            age_subset = chronological_ages[sample_idx]
            x_explain = np.column_stack([x_explain, age_subset])
            if include_status and status is not None:
                status_subset = status[sample_idx]
                x_explain = np.column_stack([x_explain, status_subset])

    # ✅ SAFER: TreeExplainer for tree models
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_explain)

        # Plot
        plt.figure()
        shap.summary_plot(shap_values, features=x_explain, feature_names=feature_list, show=False)
        plt.title(f"SHAP Summary for {target_name}")
        plt.savefig(f"{outdir}/shap_bioage_summary.png", bbox_inches="tight")
        plt.close()

        return shap_values, x_explain

    except Exception as e:
        print(f"❌ Residual SHAP crashed: {e}")
        return None, x_explain

def plot_prediction_spread(y_true, y_pred, outpath="results/prediction_spread.png"):
    df = pd.DataFrame({'true_age': y_true, 'pred_age': y_pred})
    df['age_bin'] = df['true_age'].round().astype(int)
    grouped = df.groupby('age_bin')['pred_age'].agg(
        p01=lambda x: np.percentile(x, 1),
        p10=lambda x: np.percentile(x, 10),
        p25=lambda x: np.percentile(x, 25),
        p50="median",
        p75=lambda x: np.percentile(x, 75),
        p90=lambda x: np.percentile(x, 90),
        p99=lambda x: np.percentile(x, 99),
        count="count"
    ).reset_index()

    plt.figure(figsize=(10, 6))
    plt.plot(grouped['age_bin'], grouped['p50'], label="Median", color="black")
    plt.fill_between(grouped['age_bin'], grouped['p25'], grouped['p75'], alpha=0.3, label="25–75%")
    plt.fill_between(grouped['age_bin'], grouped['p10'], grouped['p90'], alpha=0.2, label="10–90%")
    plt.fill_between(grouped['age_bin'], grouped['p01'], grouped['p99'], alpha=0.1, label="1–99%")
    plt.xlabel("True Age (binned)")
    plt.ylabel("Predicted Age")
    plt.title("Distribution of Predicted Ages by True Age")
    plt.legend()
    plt.grid(True)
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()

def plot_prediction_diagnostics(model, x_tensor, y_true, feature_names, outdir):
    os.makedirs(outdir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        y_pred = model(x_tensor).squeeze().numpy()

    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.5)
    sns.regplot(x=y_true, y=y_pred, scatter=False, color="red")
    plt.xlabel("True Age")
    plt.ylabel("Predicted Age")
    plt.title("Predicted vs True Age")
    plt.savefig(f"{outdir}/predicted_vs_true.png", bbox_inches="tight")
    plt.close()
    plot_prediction_spread(y_true, y_pred, outpath=f"{outdir}/prediction_spread.png")
    return y_pred

def explain_prediction_shap(model, x_tensor, feature_names, outdir):
    x_explain = x_tensor[:300]
    x_background = x_tensor[:100]
    wrapped_model = WrappedModel(model)

    x_explain_np = x_explain.numpy()
    x_background_np = x_background.numpy()

    def predict_fn(x_np):
        with torch.no_grad():
            x_tensor = torch.tensor(x_np, dtype=torch.float32)
            return wrapped_model(x_tensor).detach().numpy()

    explainer = shap.KernelExplainer(predict_fn, x_background_np)
    shap_values = explainer.shap_values(x_explain_np, nsamples=100)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    if shap_values.shape[-1] == 1:
        shap_values = shap_values.squeeze(-1)
    if shap_values.shape != x_explain_np.shape:
        raise ValueError(f"SHAP shape mismatch: expected {x_explain_np.shape}, got {shap_values.shape}")

    shap.summary_plot(shap_values, features=x_explain_np, feature_names=feature_names, show=False)
    plt.title("SHAP Summary Plot")
    plt.savefig(f"{outdir}/shap_summary.png", bbox_inches="tight")
    plt.close()

    for i, name in enumerate(feature_names):
        try:
            print(f"Plotting SHAP for feature: {name}")
            shap.dependence_plot(i, shap_values, x_explain_np, feature_names=feature_names, show=False)
            plt.title(f"SHAP Feature: {name}")
            plt.savefig(f"{outdir}/shap_feature_{name}.png", bbox_inches="tight")
            plt.close()
        except Exception as e:
            print(f"⚠️ Skipping feature '{name}' due to plotting error: {e}")    

    return shap_values, x_explain_np

def generate_html_report(feature_names, outdir):
    with open(f"{outdir}/report.html", "w") as f:
        f.write("<html><head><title>SHAP & Regression Report</title></head><body>\n")
        f.write("<h1>Predicted vs True Age</h1>\n")
        f.write('<img src="predicted_vs_true.png" width="600"><br><hr>\n')
        f.write('<h1>Prediction Spread by True Age</h1>\n')
        f.write('<img src="prediction_spread.png" width="600"><br><hr>\n')
        f.write("<h1>SHAP Summary Plot</h1>\n")
        f.write('<img src="shap_summary.png" width="600"><br><hr>\n')
        f.write("<h1>Feature SHAP Effects</h1>\n")
        for name in feature_names:
            f.write(f"<h2>{name}</h2>\n")
            f.write(f'<img src="shap_feature_{name}.png" width="600"><br>\n')
        f.write("<h1>SHAP Residual Explanation</h1>\n")
        f.write('<img src="shap_residual_summary.png" width="600"><br><hr>\n')
        f.write('<h1>Residual vs Prediction SHAP Correlation</h1>\n')
        for name in feature_names:
            f.write(f"<h2>{name}</h2>\n")
            f.write(f'<img src="shap_resid_vs_pred_{name}.png" width="600"><br>\n')
        f.write("</body></html>")
    print(f"✅ SHAP report complete. View it in: {outdir}/report.html")

def explain_model_with_shap(model, x_tensor, y_true, feature_names, outdir="results", model_name="amoris",
                           big_net_mode="residuals", y_bioage=None, status=None, include_status=False,
                           include_age_in_bignet=False, chronological_ages=None):
    """
    Main function to analyze model with SHAP.
    
    Parameters:
    -----------
    big_net_mode : str
        "residuals" : Train big net on prediction errors (default)
        "kdm_advance" : Train big net on KDM advance values
        "phenoage_advance" : Train big net on PhenoAge advance values
    y_bioage : array-like, optional
        Biological age values needed for kdm_advance or phenoage_advance modes
    status : array-like, optional
        Mortality status, used with phenoage_advance and include_status=True
    include_status : bool
        Include status as input feature for phenoage_advance
    """
    y_pred = plot_prediction_diagnostics(model, x_tensor, y_true, feature_names, outdir)
    shap_pred_values, x_explain_np = explain_prediction_shap(model, x_tensor, feature_names, outdir)
    shap_resid_values, _ = explain_residuals(
        x_df=pd.DataFrame(x_tensor.numpy(), columns=feature_names),
        y_true=y_true,
        y_pred=y_pred,
        feature_names=feature_names,
        outdir=outdir,
        x_explain=x_explain_np,
        big_net_mode=big_net_mode,
        y_bioage=y_bioage,
        status=status,
        include_status=include_status,
        include_age_in_bignet=include_age_in_bignet,
        chronological_ages=chronological_ages
    )
    plot_shap_residual_vs_prediction(
        shap_pred_values=shap_pred_values,
        shap_resid_values=shap_resid_values,
        feature_names=feature_names,
        outpath=outdir
    )
    generate_html_report(feature_names, outdir)
