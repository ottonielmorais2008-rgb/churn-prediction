"""Shared fixtures for churn pipeline tests."""

from __future__ import annotations

import pandas as pd
import pytest


def _row(**overrides):
    base = {
        "customerID": "0000-XXXX",
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 50.0,
        "TotalCharges": 600.0,
        "Churn": "No",
    }
    base.update(overrides)
    return base


@pytest.fixture
def messy_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _row(customerID="a", tenure=1, TotalCharges="29.85", Churn="No"),
            _row(
                customerID="b",
                tenure=0,
                TotalCharges=" ",
                MonthlyCharges=20.25,
                PhoneService="No",
                MultipleLines="No phone service",
                InternetService="No",
                OnlineSecurity="No internet service",
                OnlineBackup="No internet service",
                DeviceProtection="No internet service",
                TechSupport="No internet service",
                StreamingTV="No internet service",
                StreamingMovies="No internet service",
                Contract="Two year",
                Churn="No",
            ),
            _row(
                customerID="c",
                SeniorCitizen=1,
                InternetService="Fiber optic",
                Contract="Month-to-month",
                MonthlyCharges=90.0,
                TotalCharges="1080",
                Churn="Yes",
            ),
        ]
    )


@pytest.fixture
def sample_df() -> pd.DataFrame:
    genders = ["Female", "Male"]
    contracts = ["Month-to-month", "One year", "Two year"]
    internet = ["DSL", "Fiber optic", "No"]
    payments = [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    rows = []
    for i in range(80):
        has_internet = internet[i % 3]
        has_phone = "Yes" if i % 5 else "No"
        yes_no_net = "No internet service" if has_internet == "No" else ("Yes" if i % 2 == 0 else "No")
        rows.append(
            _row(
                customerID=f"id-{i:03d}",
                gender=genders[i % 2],
                SeniorCitizen=i % 4 == 0,
                Partner="Yes" if i % 3 == 0 else "No",
                Dependents="Yes" if i % 5 == 0 else "No",
                tenure=0 if i == 1 else (i % 72),
                PhoneService=has_phone,
                MultipleLines="No phone service" if has_phone == "No" else ("Yes" if i % 2 else "No"),
                InternetService=has_internet,
                OnlineSecurity=yes_no_net,
                OnlineBackup=yes_no_net,
                DeviceProtection=yes_no_net,
                TechSupport=yes_no_net,
                StreamingTV=yes_no_net,
                StreamingMovies=yes_no_net,
                Contract=contracts[i % 3],
                PaperlessBilling="Yes" if i % 2 else "No",
                PaymentMethod=payments[i % 4],
                MonthlyCharges=20 + (i % 80),
                TotalCharges=" " if i == 1 else (20 + (i % 80)) * max(i % 72, 1),
                Churn="Yes" if i % 4 == 0 else "No",
            )
        )
    return pd.DataFrame(rows)
