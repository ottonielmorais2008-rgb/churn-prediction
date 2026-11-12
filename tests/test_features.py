from churn.data import clean
from churn.features import add_features


def test_avg_monthly_spend_uses_ratio_or_monthly_for_new_customers(messy_df):
    out = add_features(clean(messy_df))
    assert out.loc[0, "avg_monthly_spend"] == 29.85
    # tenure 0 and blank TotalCharges should fall back to MonthlyCharges
    assert out.loc[1, "avg_monthly_spend"] == 20.25
    assert out.loc[2, "avg_monthly_spend"] == 90.0


def test_num_services_counts_addons_phone_and_internet(messy_df):
    out = add_features(clean(messy_df))
    # row 0: phone + DSL + OnlineBackup
    assert int(out.loc[0, "num_services"]) == 3
    # row 1: no phone, no internet, no add-ons
    assert int(out.loc[1, "num_services"]) == 0
    # row 2: phone + fiber, no add-ons marked Yes in the fixture except none
    assert int(out.loc[2, "num_services"]) >= 2


def test_tenure_group_bins(messy_df):
    out = add_features(clean(messy_df))
    assert out.loc[0, "tenure_group"] == "0-12"
    assert out.loc[1, "tenure_group"] == "0-12"
