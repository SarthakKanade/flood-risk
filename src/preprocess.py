"""Preprocessing module for FloodSense: Regional Flood Risk Analysis.

Provides reproducible functions for:
1. Loading processed data.
2. Train/test splitting (80/20) strictly before any fitting to prevent data leakage.
3. Computing quantile-based risk tier cutoffs from the training target.
4. Optional feature engineering of domain aggregates.
5. Scikit-learn pipelines with StandardScaler for models requiring scaling.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Feature domain groups
ENV_INDICATORS = [
    "MonsoonIntensity",
    "TopographyDrainage",
    "Deforestation",
    "ClimateChange",
    "CoastalVulnerability",
    "Landslides",
    "Watersheds",
    "WetlandLoss"
]

INFRA_GOV_INDICATORS = [
    "RiverManagement",
    "Urbanization",
    "DamsQuality",
    "Siltation",
    "AgriculturalPractices",
    "Encroachments",
    "IneffectiveDisasterPreparedness",
    "DrainageSystems",
    "DeterioratingInfrastructure",
    "PopulationScore",
    "InadequatePlanning",
    "PoliticalFactors"
]

ALL_20_INDICATORS = ENV_INDICATORS + INFRA_GOV_INDICATORS


def load_data(filepath="flood-risk/data/processed/sample_100k.csv"):
    """Load the processed dataset."""
    if not os.path.exists(filepath):
        # Fallback if called from another working directory
        alt_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "sample_100k.csv")
        if os.path.exists(alt_path):
            filepath = alt_path
        else:
            raise FileNotFoundError(f"Processed data file not found at: {filepath}")
    return pd.read_csv(filepath)


def split_data(df, test_size=0.20, random_state=42):
    """Split data into train (80%) and test (20%) sets prior to any fitting.
    
    Prevents data leakage by ensuring all subsequent scalers, imputers, 
    and tier cutoffs are learned strictly from the training split.
    """
    X = df.drop(columns=["FloodProbability"])
    y = df["FloodProbability"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def compute_tier_cutoffs(y_train, save_path="flood-risk/models/tier_cutoffs.json"):
    """Calculate 3-class risk tier cutoffs from training target percentiles.
    
    Bottom third (0% - 33.33%): Low Risk
    Middle third (33.33% - 66.67%): Medium Risk
    Top third (66.67% - 100%): High Risk
    
    Returns:
        dict: {'q33': float, 'q67': float}
    """
    q33 = float(np.percentile(y_train, 100.0 / 3.0))
    q67 = float(np.percentile(y_train, 200.0 / 3.0))
    cutoffs = {
        "q33": q33,
        "q67": q67,
        "labels": ["Low", "Medium", "High"],
        "description": "Quantile-based regional risk tiers calculated on training split."
    }
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(cutoffs, f, indent=4)
        print(f"Saved tier cutoffs to {save_path}: Low <= {q33:.4f} < Medium <= {q67:.4f} < High")
        
    return cutoffs


def assign_risk_tiers(y_series, cutoffs):
    """Assign risk tier labels (Low, Medium, High) given probability values and cutoffs."""
    q33 = cutoffs["q33"]
    q67 = cutoffs["q67"]
    
    tiers = pd.Series(index=y_series.index, dtype="object")
    tiers[y_series <= q33] = "Low"
    tiers[(y_series > q33) & (y_series <= q67)] = "Medium"
    tiers[y_series > q67] = "High"
    return tiers


def add_engineered_features(df_features):
    """Add 3 simple domain aggregate features.
    
    1. indicator_mean: average of all 20 indicators (overall vulnerability burden).
    2. env_mean: average of environmental and hydrological indicators.
    3. infra_mean: average of infrastructure and governance indicators.
    """
    df_out = df_features.copy()
    
    # Check that required columns exist
    cols = df_out.columns
    existing_all = [c for c in ALL_20_INDICATORS if c in cols]
    existing_env = [c for c in ENV_INDICATORS if c in cols]
    existing_infra = [c for c in INFRA_GOV_INDICATORS if c in cols]
    
    df_out["indicator_mean"] = df_out[existing_all].mean(axis=1)
    df_out["env_mean"] = df_out[existing_env].mean(axis=1)
    df_out["infra_mean"] = df_out[existing_infra].mean(axis=1)
    
    return df_out


def build_pipeline(model, scale_features=True, feature_names=None):
    """Build a scikit-learn Pipeline ensuring data leakage prevention.
    
    If scale_features=True, prepends a StandardScaler that fits only on train folds.
    If scale_features=False, passes features directly to model (for tree ensembles).
    """
    if scale_features:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("regressor", model)
        ])
    else:
        return Pipeline([
            ("regressor", model)
        ])
