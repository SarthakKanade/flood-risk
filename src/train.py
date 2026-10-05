"""Model training, experiment comparison, and artifact serialization for FloodSense.

Executes:
1. 5-fold cross-validation across all baseline and candidate models.
2. Experiments:
   (a) Scaled vs. Unscaled inputs
   (b) Original 20 features vs. Original + Engineered domain aggregates
   (c) Hyperparameter tuning on top candidate models
3. Single holdout test set evaluation.
4. Export of reports/results/model_comparison.csv.
5. Final pipeline selection and serialization to models/final_model.joblib.
"""

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_validate, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.preprocess import (
    load_data,
    split_data,
    compute_tier_cutoffs,
    add_engineered_features,
    build_pipeline
)

RESULTS_DIR = "flood-risk/reports/results"
MODELS_DIR = "flood-risk/models"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def evaluate_model_cv(pipeline, X_train, y_train, cv=5):
    """Run K-Fold cross validation and compute R2, RMSE, and MAE."""
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    scoring = {
        "r2": "r2",
        "neg_rmse": "neg_root_mean_squared_error",
        "neg_mae": "neg_mean_absolute_error"
    }
    
    t0 = time.time()
    scores = cross_validate(pipeline, X_train, y_train, cv=kf, scoring=scoring, n_jobs=-1)
    fit_time = time.time() - t0
    
    return {
        "cv_r2_mean": float(np.mean(scores["test_r2"])),
        "cv_r2_std": float(np.std(scores["test_r2"])),
        "cv_rmse_mean": float(-np.mean(scores["test_neg_rmse"])),
        "cv_rmse_std": float(np.std(scores["test_neg_rmse"])),
        "cv_mae_mean": float(-np.mean(scores["test_neg_mae"])),
        "cv_mae_std": float(np.std(scores["test_neg_mae"])),
        "train_time_sec": round(fit_time, 2)
    }


def evaluate_model_test(pipeline, X_train, y_train, X_test, y_test):
    """Fit pipeline on full training set and evaluate on unseen test set."""
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    r2 = float(r2_score(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))
    
    return {
        "test_r2": r2,
        "test_rmse": rmse,
        "test_mae": mae,
        "fitted_pipeline": pipeline
    }


def run_all_experiments():
    """Execute all modeling experiments and return comparison DataFrame."""
    print("=" * 60)
    print("Starting FloodSense Model Training & Comparison")
    print("=" * 60)
    
    # 1. Load and Split
    df = load_data("flood-risk/data/processed/sample_100k.csv")
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.20, random_state=42)
    cutoffs = compute_tier_cutoffs(y_train, save_path=os.path.join(MODELS_DIR, "tier_cutoffs.json"))
    
    # Engineered feature sets
    X_train_eng = add_engineered_features(X_train)
    X_test_eng = add_engineered_features(X_test)
    
    experiments = []
    trained_pipelines = {}

    # Define base models
    models_config = [
        # (Model Name, Experiment Category, Model Object, Scale Flag, Feature Set X_tr, Feature Set X_te)
        ("Linear Regression (Unscaled, 20 Feat)", "Scaling Experiment", LinearRegression(), False, X_train, X_test),
        ("Linear Regression (Scaled, 20 Feat)", "Baseline", LinearRegression(), True, X_train, X_test),
        ("Ridge Regression (Scaled, default alpha=1.0)", "Regularization", Ridge(alpha=1.0, random_state=42), True, X_train, X_test),
        ("Ridge Regression (Unscaled, default alpha=1.0)", "Scaling Experiment", Ridge(alpha=1.0, random_state=42), False, X_train, X_test),
        ("HistGradientBoosting (Default, 20 Feat)", "Ensemble Non-Linear", HistGradientBoostingRegressor(max_iter=100, random_state=42), False, X_train, X_test),
        ("HistGradientBoosting (150 iters, lr=0.08)", "Ensemble Non-Linear", HistGradientBoostingRegressor(max_iter=150, learning_rate=0.08, random_state=42), False, X_train, X_test),
        ("Random Forest (50 Trees, max_depth=10)", "Ensemble Non-Linear", RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1), False, X_train, X_test),
        ("MLP Neural Net (Scaled, 32-16 layers)", "Neural Network", MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=35, early_stopping=True, random_state=42), True, X_train, X_test),
        
        # Feature engineering experiments
        ("Linear Regression (Scaled, 20 + 3 Aggregates)", "Feature Engineering", LinearRegression(), True, X_train_eng, X_test_eng),
        ("HistGradientBoosting (20 + 3 Aggregates)", "Feature Engineering", HistGradientBoostingRegressor(max_iter=100, random_state=42), False, X_train_eng, X_test_eng),
        ("Ridge Regression (Scaled, 20 + 3 Aggregates)", "Feature Engineering", Ridge(alpha=1.0, random_state=42), True, X_train_eng, X_test_eng),
    ]

    for name, category, model_obj, scale_flag, x_tr, x_te in models_config:
        print(f"--> Evaluating: {name}...")
        pipe = build_pipeline(model_obj, scale_features=scale_flag)
        
        # 5-Fold Cross Validation
        cv_res = evaluate_model_cv(pipe, x_tr, y_train, cv=5)
        
        # Test Set Evaluation
        test_res = evaluate_model_test(pipe, x_tr, y_train, x_te, y_test)
        trained_pipelines[name] = test_res["fitted_pipeline"]
        
        experiments.append({
            "Model": name,
            "Category": category,
            "Scaled": scale_flag,
            "Features": "23 (Engineered)" if "Aggregates" in name else "20 (Original)",
            "CV R2 Mean": round(cv_res["cv_r2_mean"], 5),
            "CV R2 Std": round(cv_res["cv_r2_std"], 5),
            "CV RMSE": round(cv_res["cv_rmse_mean"], 5),
            "CV MAE": round(cv_res["cv_mae_mean"], 5),
            "Test R2": round(test_res["test_r2"], 5),
            "Test RMSE": round(test_res["test_rmse"], 5),
            "Test MAE": round(test_res["test_mae"], 5),
            "CV Train Time (s)": cv_res["train_time_sec"]
        })

    # Hyperparameter Tuning on Top 2 Candidates: Ridge and HistGradientBoosting
    print("\n--> Running Hyperparameter Tuning: Ridge Regression...")
    ridge_pipe = build_pipeline(Ridge(random_state=42), scale_features=True)
    param_dist_ridge = {
        "regressor__alpha": np.logspace(-2, 3, 20)
    }
    rsearch_ridge = RandomizedSearchCV(
        ridge_pipe, param_distributions=param_dist_ridge, n_iter=10,
        cv=5, scoring="r2", random_state=42, n_jobs=-1
    )
    t0 = time.time()
    rsearch_ridge.fit(X_train, y_train)
    tune_time_ridge = time.time() - t0
    best_ridge_pipe = rsearch_ridge.best_estimator_
    
    cv_res_ridge = evaluate_model_cv(best_ridge_pipe, X_train, y_train, cv=5)
    test_res_ridge = evaluate_model_test(best_ridge_pipe, X_train, y_train, X_test, y_test)
    
    trained_pipelines[f"Tuned Ridge (alpha={rsearch_ridge.best_params_['regressor__alpha']:.3f})"] = test_res_ridge["fitted_pipeline"]
    experiments.append({
        "Model": f"Tuned Ridge (alpha={rsearch_ridge.best_params_['regressor__alpha']:.3f})",
        "Category": "Hyperparameter Tuning",
        "Scaled": True,
        "Features": "20 (Original)",
        "CV R2 Mean": round(cv_res_ridge["cv_r2_mean"], 5),
        "CV R2 Std": round(cv_res_ridge["cv_r2_std"], 5),
        "CV RMSE": round(cv_res_ridge["cv_rmse_mean"], 5),
        "CV MAE": round(cv_res_ridge["cv_mae_mean"], 5),
        "Test R2": round(test_res_ridge["test_r2"], 5),
        "Test RMSE": round(test_res_ridge["test_rmse"], 5),
        "Test MAE": round(test_res_ridge["test_mae"], 5),
        "CV Train Time (s)": round(tune_time_ridge, 2)
    })

    print("--> Running Hyperparameter Tuning: HistGradientBoosting...")
    hgb_pipe = build_pipeline(HistGradientBoostingRegressor(random_state=42), scale_features=False)
    param_dist_hgb = {
        "regressor__max_iter": [100, 150],
        "regressor__learning_rate": [0.05, 0.1, 0.15],
        "regressor__max_leaf_nodes": [31, 50],
        "regressor__min_samples_leaf": [20, 50]
    }
    rsearch_hgb = RandomizedSearchCV(
        hgb_pipe, param_distributions=param_dist_hgb, n_iter=6,
        cv=5, scoring="r2", random_state=42, n_jobs=-1
    )
    t0 = time.time()
    rsearch_hgb.fit(X_train, y_train)
    tune_time_hgb = time.time() - t0
    best_hgb_pipe = rsearch_hgb.best_estimator_
    
    cv_res_hgb = evaluate_model_cv(best_hgb_pipe, X_train, y_train, cv=5)
    test_res_hgb = evaluate_model_test(best_hgb_pipe, X_train, y_train, X_test, y_test)
    
    trained_pipelines["Tuned HistGradientBoosting"] = test_res_hgb["fitted_pipeline"]
    experiments.append({
        "Model": "Tuned HistGradientBoosting",
        "Category": "Hyperparameter Tuning",
        "Scaled": False,
        "Features": "20 (Original)",
        "CV R2 Mean": round(cv_res_hgb["cv_r2_mean"], 5),
        "CV R2 Std": round(cv_res_hgb["cv_r2_std"], 5),
        "CV RMSE": round(cv_res_hgb["cv_rmse_mean"], 5),
        "CV MAE": round(cv_res_hgb["cv_mae_mean"], 5),
        "Test R2": round(test_res_hgb["test_r2"], 5),
        "Test RMSE": round(test_res_hgb["test_rmse"], 5),
        "Test MAE": round(test_res_hgb["test_mae"], 5),
        "CV Train Time (s)": round(tune_time_hgb, 2)
    })

    # Convert to DataFrame
    df_results = pd.DataFrame(experiments)
    results_csv_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    df_results.to_csv(results_csv_path, index=False)
    print(f"\n--> Model comparison saved to: {results_csv_path}")
    print("\nComparison Table:")
    print(df_results.to_string())

    # Selection of Final Model:
    # Compare CV R2 scores. As expected from problem definition:
    # "Because Linear Regression is already strong, expect only SMALL gaps between models. That is a valid, honest result...
    # If differences are negligible, prefer the simpler model, and explain why."
    
    print("\n" + "=" * 60)
    print("Final Model Selection Rationale:")
    print("=" * 60)
    # Check top CV R2
    best_cv_idx = df_results["CV R2 Mean"].idxmax()
    best_model_name = df_results.loc[best_cv_idx, "Model"]
    best_cv_score = df_results.loc[best_cv_idx, "CV R2 Mean"]
    
    lr_cv_score = df_results.loc[df_results["Model"] == "Linear Regression (Scaled, 20 Feat)", "CV R2 Mean"].values[0]
    score_diff = best_cv_score - lr_cv_score
    print(f"Top Performer by CV R2: {best_model_name} (CV R2 = {best_cv_score:.5f})")
    print(f"Linear Regression (Scaled): CV R2 = {lr_cv_score:.5f}")
    print(f"Performance Gap: {score_diff:.5f} ({score_diff*100:.3f}% difference)")

    # If score difference is < 0.005 (0.5%), select Ridge / Linear Regression for parsimony, explainability, and speed!
    if score_diff < 0.005:
        selected_model_name = "Ridge Regression (Scaled, default alpha=1.0)"
        final_pipeline = trained_pipelines[selected_model_name]
        selection_reason = (
            "Occam's Razor & Interpretability: Complex models showed negligible improvement over linear models "
            f"(gap of {score_diff:.5f} in R2). Ridge Regression with StandardScaler is chosen for maximum transparency "
            "and direct coefficient interpretability."
        )
    else:
        selected_model_name = best_model_name
        selection_reason = f"Chosen based on strictly superior cross-validation score: {best_cv_score:.5f} R2 (+{score_diff*100:.2f}% gain)."
        if "Aggregates" in selected_model_name:
            from sklearn.preprocessing import FunctionTransformer
            final_pipeline = Pipeline([
                ("feature_engineering", FunctionTransformer(add_engineered_features)),
                ("regressor", HistGradientBoostingRegressor(max_iter=100, random_state=42))
            ])
            final_pipeline.fit(X_train, y_train)
        else:
            final_pipeline = trained_pipelines[selected_model_name]

    print(f"\nFINAL MODEL SELECTED: {selected_model_name}")
    print(f"RATIONALE: {selection_reason}")

    final_model_path = os.path.join(MODELS_DIR, "final_model.joblib")
    joblib.dump(final_pipeline, final_model_path)
    print(f"\nSaved final model pipeline to: {final_model_path}")
    
    return df_results, selected_model_name, selection_reason


if __name__ == "__main__":
    run_all_experiments()
