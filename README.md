# Telco customer churn prediction

A small end-to-end project that predicts whether a telecom customer is likely to leave. It trains a logistic regression baseline and a random forest on a public dataset, writes a metrics report, and ships a [live demo](https://ottonielmorais2008-rgb.github.io/churn-prediction/) that runs the logistic model entirely in the browser.

[![CI](https://github.com/ottonielmorais2008-rgb/churn-prediction/actions/workflows/ci.yml/badge.svg)](https://github.com/ottonielmorais2008-rgb/churn-prediction/actions/workflows/ci.yml)

**Live demo:** [https://ottonielmorais2008-rgb.github.io/churn-prediction/](https://ottonielmorais2008-rgb.github.io/churn-prediction/)

## Problem

Customer churn is costly, and a model that ranks who is most likely to leave can help a retention team focus its effort. This project uses the well-known IBM Telco Customer Churn table: each row is one customer, with contract and service details, and a yes/no label for whether they left.

## Dataset and licence

The training data is the [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d) dataset (`data/Telco-Customer-Churn.csv` in that repository). The repository is licensed under the [Apache License 2.0](https://github.com/IBM/telco-customer-churn-on-icp4d/blob/master/LICENSE). The CSV is downloaded at runtime and is not copied into this repository. See `NOTICE` for attribution.

After cleaning there are 7,043 customers, of whom 26.5% churned.

## Approach

1. Clean the table: drop `customerID`, coerce `TotalCharges` to a number (blank values become 0), map `Churn` to 0/1, and collapse `"No internet service"` / `"No phone service"` to `"No"`.
2. Add three features: average monthly spend (`TotalCharges / tenure`, or `MonthlyCharges` when tenure is 0), a count of active services, and a tenure bucket (`0-12`, `13-24`, `25-48`, `49-72`).
3. Split 80/20, stratified, with `random_state=42`.
4. Scale numeric columns, one-hot encode categoricals, and fit two models with `class_weight="balanced"`:
   - logistic regression (baseline, also used in the demo)
   - random forest (80 trees, `max_depth=8`)
5. Report ROC-AUC, precision, recall, F1, accuracy, and a confusion matrix on the test set at a 0.5 threshold. Export the logistic coefficients and scaler so the static page can score new profiles in JavaScript.

## Results

Numbers below come from one training run on the held-out test set of 1,409 customers (5,634 used for training). They are also stored in `reports/metrics.json` and `reports/REPORT.md`.

| Model | ROC-AUC | Precision | Recall | F1 | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Logistic regression | 0.842 | 0.497 | 0.791 | 0.611 | 0.732 |
| Random forest | 0.845 | 0.529 | 0.805 | 0.638 | 0.758 |

Confusion matrices:

- Logistic regression: TN 736, FP 299, FN 78, TP 296
- Random forest: TN 767, FP 268, FN 73, TP 301

The strongest logistic weights match the usual story in this dataset: longer tenure and a two-year contract lower the predicted risk; fibre-optic internet and a month-to-month contract raise it. Charts are in `reports/` and on the demo page.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
PYTHONPATH=src python -m churn.train
```

The last command downloads the dataset (if needed), retrains both models, and refreshes `reports/` plus `docs/model.json`. To view the demo on your machine, serve the `docs` folder over HTTP, for example:

```bash
python -m http.server --directory docs 8000
```

Then open `http://localhost:8000`. Opening the HTML file directly may block loading `model.json`.

## Limitations

- This is a snapshot of historical, fictional customers. It does not track people over time, so it cannot separate cause from correlation.
- The 0.5 threshold with balanced class weights favours recall over precision. About half of the customers flagged by the logistic model did not actually churn (precision 0.50). A retention team would need to pick a threshold that matches the cost of a false alarm.
- The live demo only runs the logistic model, and fields not shown in the form are filled with training-set defaults (or `monthly charges × tenure` for total charges). That is good enough for a walkthrough, not for production scoring.
- Average monthly spend is close to `MonthlyCharges`, and the tenure bucket overlaps with raw tenure, so some coefficients are harder to read on their own.

## What went wrong

- `TotalCharges` arrived as text. New customers with tenure 0 had a blank in that column, which pandas would otherwise turn into NaN and drop or break the scaler. Those blanks are now filled with 0, and average spend falls back to `MonthlyCharges`.
- Several add-on columns used `"No internet service"` or `"No phone service"` as a third level, which duplicated `InternetService` and `PhoneService`. Collapsing those levels to `"No"` removed an artificial split.
- The random forest is only slightly ahead of logistic regression on ROC-AUC (0.845 vs 0.842). The extra model is useful as a check, not because it transforms the result.

## Licence

The code in this repository is under the MIT License. The dataset remains under the Apache License 2.0 of its upstream source.
