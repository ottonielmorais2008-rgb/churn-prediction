"""Download and clean the IBM Telco Customer Churn dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

DATASET_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
DATASET_SOURCE = "https://github.com/IBM/telco-customer-churn-on-icp4d"
DATASET_LICENSE = "Apache License 2.0"
DATASET_NAME = "IBM Telco Customer Churn"

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"

NO_SERVICE_VALUES = ("No internet service", "No phone service")


def download_dataset(dest: Path | None = None, url: str = DATASET_URL) -> Path:
    """Download the CSV if it is not already present. Returns the local path."""
    dest = Path(dest) if dest is not None else RAW_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    df = pd.read_csv(url)
    if df.empty or "Churn" not in df.columns:
        raise ValueError(f"Downloaded file from {url} is not the expected churn dataset")
    df.to_csv(dest, index=False)
    return dest


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """Load a raw Telco churn CSV, downloading the official file when needed."""
    if path is None:
        path = download_dataset()
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise types, fill blank TotalCharges, and collapse 'no service' levels."""
    out = df.copy()
    if "customerID" in out.columns:
        out = out.drop(columns=["customerID"])
    if "TotalCharges" in out.columns:
        out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce").fillna(0.0)
    if "Churn" in out.columns and not pd.api.types.is_integer_dtype(out["Churn"]):
        mapped = out["Churn"].map({"Yes": 1, "No": 0, 1: 1, 0: 0, True: 1, False: 0})
        if mapped.isna().any():
            raise ValueError("Churn column contains values other than Yes/No")
        out["Churn"] = mapped.astype(int)
    if "SeniorCitizen" in out.columns:
        out["SeniorCitizen"] = out["SeniorCitizen"].astype(int)
    object_cols = out.select_dtypes(include=["object", "string"]).columns
    for col in object_cols:
        out[col] = out[col].replace({value: "No" for value in NO_SERVICE_VALUES})
    return out
