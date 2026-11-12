"""A small set of engineered features for the Telco churn dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd

TARGET = "Churn"

ADDON_COLS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

NUMERIC_FEATURES = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "SeniorCitizen",
    "avg_monthly_spend",
    "num_services",
]

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "tenure_group",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add average spend, service count, and a tenure bucket."""
    out = df.copy()
    tenure = out["tenure"].astype(float)
    total = out["TotalCharges"].astype(float)
    monthly = out["MonthlyCharges"].astype(float)
    out["avg_monthly_spend"] = np.where(tenure > 0, total / tenure, monthly)

    service_flags = []
    for col in ADDON_COLS:
        if col in out.columns:
            service_flags.append((out[col] == "Yes").astype(int))
    if "PhoneService" in out.columns:
        service_flags.append((out["PhoneService"] == "Yes").astype(int))
    if "InternetService" in out.columns:
        service_flags.append((out["InternetService"] != "No").astype(int))
    out["num_services"] = sum(service_flags) if service_flags else 0

    out["tenure_group"] = pd.cut(
        out["tenure"],
        bins=[-0.1, 12, 24, 48, 72],
        labels=["0-12", "13-24", "25-48", "49-72"],
    ).astype(str)
    return out


def model_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None]:
    """Return the model columns and, when present, the target."""
    missing = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns after feature engineering: {missing}")
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()
    y = df[TARGET].astype(int) if TARGET in df.columns else None
    return X, y
