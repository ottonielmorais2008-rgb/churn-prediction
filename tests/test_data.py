from churn.data import clean


def test_clean_drops_customer_id_and_maps_churn(messy_df):
    out = clean(messy_df)
    assert "customerID" not in out.columns
    assert list(out["Churn"]) == [0, 0, 1]


def test_clean_coerces_blank_total_charges(messy_df):
    out = clean(messy_df)
    assert out.loc[1, "TotalCharges"] == 0.0
    assert out["TotalCharges"].dtype.kind == "f"
    assert out.loc[0, "TotalCharges"] == 29.85


def test_clean_collapses_no_service_levels(messy_df):
    out = clean(messy_df)
    assert out.loc[1, "MultipleLines"] == "No"
    assert out.loc[1, "OnlineSecurity"] == "No"
    assert "No internet service" not in out["OnlineSecurity"].tolist()
    assert "No phone service" not in out["MultipleLines"].tolist()
