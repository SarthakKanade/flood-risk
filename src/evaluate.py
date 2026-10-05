"""Comprehensive evaluation module for FloodSense.

Computes:
1. Test regression metrics (R2, RMSE, MAE).
2. Risk tier classification metrics (Confusion matrix, precision, recall, F1, macro-F1).
3. Residual & error diagnostic figures saved to reports/figures/.
4. Top 10 largest prediction errors inspection.
5. Permutation feature importance on the test set.
6. Exports all numerical results to models/metrics.json.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    confusion_matrix,
    classification_report,
    f1_score
)
from sklearn.inspection import permutation_importance

# Ensure path is set
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocess import load_data, split_data, assign_risk_tiers, add_engineered_features

REPORTS_FIG_DIR = "flood-risk/reports/figures"
MODELS_DIR = "flood-risk/models"
os.makedirs(REPORTS_FIG_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")


def run_evaluation():
    print("=" * 60)
    print("Starting FloodSense Model Evaluation & Diagnostics")
    print("=" * 60)

    # 1. Load Data, Model, and Cutoffs
    df = load_data("flood-risk/data/processed/sample_100k.csv")
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.20, random_state=42)
    
    with open(os.path.join(MODELS_DIR, "tier_cutoffs.json"), "r") as f:
        cutoffs = json.load(f)
    print(f"Loaded tier cutoffs: Low <= {cutoffs['q33']:.4f} < Medium <= {cutoffs['q67']:.4f} < High")

    model_path = os.path.join(MODELS_DIR, "final_model.joblib")
    model = joblib.load(model_path)
    print(f"Loaded final pipeline from: {model_path}")

    # 2. Regression Evaluation
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred
    abs_residuals = np.abs(residuals)

    r2 = float(r2_score(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))

    print("\n--- Regression Test Performance ---")
    print(f"Test R2 Score: {r2:.5f}")
    print(f"Test RMSE:     {rmse:.5f}")
    print(f"Test MAE:      {mae:.5f}")

    # 3. Risk Tier Classification Evaluation
    true_tiers = assign_risk_tiers(y_test, cutoffs)
    pred_tiers = assign_risk_tiers(pd.Series(y_pred, index=y_test.index), cutoffs)
    tier_labels = ["Low", "Medium", "High"]

    cm = confusion_matrix(true_tiers, pred_tiers, labels=tier_labels)
    class_rep = classification_report(true_tiers, pred_tiers, labels=tier_labels, output_dict=True)
    macro_f1 = float(f1_score(true_tiers, pred_tiers, labels=tier_labels, average="macro"))

    print("\n--- Risk Tier Classification Performance ---")
    print("Confusion Matrix (Rows: Actual, Cols: Predicted):")
    cm_df = pd.DataFrame(cm, index=[f"Actual_{l}" for l in tier_labels], columns=[f"Pred_{l}" for l in tier_labels])
    print(cm_df)
    print(f"\nMacro-F1 Score: {macro_f1:.5f}")
    print("\nClassification Report:")
    for lbl in tier_labels:
        p = class_rep[lbl]["precision"]
        r = class_rep[lbl]["recall"]
        f = class_rep[lbl]["f1-score"]
        sup = class_rep[lbl]["support"]
        print(f"  {lbl:6s} -> Precision: {p:.4f} | Recall: {r:.4f} | F1: {f:.4f} | Support: {sup}")

    # 4. Save Plots
    # Plot A: Predicted vs Actual Scatter
    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, y_pred, alpha=0.15, color="#1f77b4", s=10, edgecolor="none")
    plt.plot([0.25, 0.75], [0.25, 0.75], "r--", linewidth=2, label="Ideal 45° Line (y = x)")
    plt.axvline(cutoffs["q33"], color="gray", linestyle=":", alpha=0.7)
    plt.axvline(cutoffs["q67"], color="gray", linestyle=":", alpha=0.7)
    plt.axhline(cutoffs["q33"], color="gray", linestyle=":", alpha=0.7)
    plt.axhline(cutoffs["q67"], color="gray", linestyle=":", alpha=0.7)
    plt.title("Predicted vs. Actual Flood Probability (Test Set)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Actual Flood Probability", fontsize=11)
    plt.ylabel("Predicted Flood Probability", fontsize=11)
    plt.xlim(0.28, 0.74)
    plt.ylim(0.28, 0.74)
    plt.legend(loc="upper left")
    plt.tight_layout()
    pred_act_path = os.path.join(REPORTS_FIG_DIR, "predicted_vs_actual.png")
    plt.savefig(pred_act_path, dpi=180)
    plt.close()

    # Plot B: Residuals vs Predicted
    plt.figure(figsize=(8, 5))
    plt.scatter(y_pred, residuals, alpha=0.15, color="#2ca02c", s=10, edgecolor="none")
    plt.axhline(0, color="red", linestyle="--", linewidth=1.5)
    plt.title("Residuals vs. Predicted Values", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Flood Probability", fontsize=11)
    plt.ylabel("Residual (Actual - Predicted)", fontsize=11)
    plt.ylim(-0.15, 0.15)
    plt.tight_layout()
    res_pred_path = os.path.join(REPORTS_FIG_DIR, "residuals_vs_predicted.png")
    plt.savefig(res_pred_path, dpi=180)
    plt.close()

    # Plot C: Residual Histogram
    plt.figure(figsize=(8, 5))
    sns.histplot(residuals, kde=True, bins=50, color="#6baed6", edgecolor="white", alpha=0.8)
    plt.axvline(residuals.mean(), color="red", linestyle="--", label=f"Mean Error: {residuals.mean():.6f}")
    plt.title("Histogram of Prediction Residuals", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Residual (Actual - Predicted)", fontsize=11)
    plt.ylabel("Count", fontsize=11)
    plt.legend()
    plt.tight_layout()
    res_hist_path = os.path.join(REPORTS_FIG_DIR, "residual_histogram.png")
    plt.savefig(res_hist_path, dpi=180)
    plt.close()

    # Plot D: Confusion Matrix Heatmap
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=tier_labels, yticklabels=tier_labels, cbar=False)
    plt.title("Risk Tier Confusion Matrix", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Predicted Risk Tier", fontsize=11)
    plt.ylabel("Actual Risk Tier", fontsize=11)
    plt.tight_layout()
    cm_path = os.path.join(REPORTS_FIG_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=180)
    plt.close()

    # 5. Top 10 Largest Errors Inspection
    test_eval_df = X_test.copy()
    test_eval_df["Actual_Prob"] = y_test
    test_eval_df["Predicted_Prob"] = y_pred
    test_eval_df["Residual"] = residuals
    test_eval_df["Abs_Residual"] = abs_residuals
    test_eval_df["Actual_Tier"] = true_tiers
    test_eval_df["Predicted_Tier"] = pred_tiers

    top10_errors = test_eval_df.sort_values(by="Abs_Residual", ascending=False).head(10)
    print("\n--- Top 10 Largest Prediction Errors ---")
    cols_display = ["Actual_Prob", "Predicted_Prob", "Residual", "Actual_Tier", "Predicted_Tier", "MonsoonIntensity", "TopographyDrainage", "ClimateChange", "DamsQuality"]
    print(top10_errors[cols_display].to_string())

    # Error analysis patterns:
    # Check mean error near tier boundaries vs extreme values
    boundary_width = 0.015
    q33 = cutoffs["q33"]
    q67 = cutoffs["q67"]
    near_boundaries = ((np.abs(y_test - q33) < boundary_width) | (np.abs(y_test - q67) < boundary_width))
    extremes = (y_test < 0.40) | (y_test > 0.62)
    middle = ~near_boundaries & ~extremes

    err_boundary = abs_residuals[near_boundaries].mean()
    err_extreme = abs_residuals[extremes].mean()
    err_middle = abs_residuals[middle].mean()
    print(f"\nMean Absolute Error near Tier Boundaries (±{boundary_width}): {err_boundary:.5f} (n={near_boundaries.sum()})")
    print(f"Mean Absolute Error in Middle Safe Regions:              {err_middle:.5f} (n={middle.sum()})")
    print(f"Mean Absolute Error at Extreme Ends (<0.40 or >0.62):   {err_extreme:.5f} (n={extremes.sum()})")

    # 6. Permutation Importance
    print("\n--- Computing Permutation Feature Importance (5 repeats)... ---")
    perm_res = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1, scoring="r2")
    sorted_importances_idx = perm_res.importances_mean.argsort()[::-1]
    
    perm_df = pd.DataFrame({
        "Feature": X_test.columns[sorted_importances_idx],
        "Importance_Mean": perm_res.importances_mean[sorted_importances_idx],
        "Importance_Std": perm_res.importances_std[sorted_importances_idx]
    })
    print("\nPermutation Feature Importance Ranking (Top 10):")
    print(perm_df.head(10).to_string(index=False))

    # Plot E: Permutation Feature Importance
    plt.figure(figsize=(10, 8))
    plt.barh(perm_df["Feature"][::-1], perm_df["Importance_Mean"][::-1], xerr=perm_df["Importance_Std"][::-1], color="#1f77b4", alpha=0.85, edgecolor="black")
    plt.title("Permutation Feature Importance (Impact on Test R²)", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Decrease in Test R² when feature is randomly shuffled", fontsize=11)
    plt.grid(axis="x", linestyle="--", alpha=0.4)
    plt.tight_layout()
    perm_path = os.path.join(REPORTS_FIG_DIR, "permutation_importance.png")
    plt.savefig(perm_path, dpi=180)
    plt.close()

    # 7. Save Metrics JSON
    metrics = {
        "model_name": "HistGradientBoostingRegressor (with internal Feature Engineering)",
        "test_regression": {
            "r2": r2,
            "rmse": rmse,
            "mae": mae
        },
        "risk_tier_classification": {
            "macro_f1": macro_f1,
            "confusion_matrix": cm.tolist(),
            "tier_labels": tier_labels,
            "class_metrics": {
                lbl: {
                    "precision": round(class_rep[lbl]["precision"], 5),
                    "recall": round(class_rep[lbl]["recall"], 5),
                    "f1_score": round(class_rep[lbl]["f1-score"], 5),
                    "support": int(class_rep[lbl]["support"])
                } for lbl in tier_labels
            }
        },
        "error_analysis": {
            "max_absolute_error": float(abs_residuals.max()),
            "mean_absolute_error": float(abs_residuals.mean()),
            "mae_near_tier_boundaries": float(err_boundary),
            "mae_middle_regions": float(err_middle),
            "mae_extreme_values": float(err_extreme),
            "top10_errors_indices": top10_errors.index.tolist()
        },
        "permutation_importance": perm_df.to_dict(orient="records")
    }

    metrics_json_path = os.path.join(MODELS_DIR, "metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)
    print(f"\nSaved metrics to: {metrics_json_path}")
    print("\nEvaluation completed successfully.")
    return metrics


if __name__ == "__main__":
    run_evaluation()
