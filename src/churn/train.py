"""Train a logistic regression baseline and a random forest, then export reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.data import (
    DATASET_LICENSE,
    DATASET_NAME,
    DATASET_SOURCE,
    DATASET_URL,
    clean,
    load_raw,
)
from churn.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    add_features,
    model_matrix,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DOCS = ROOT / "docs"
DEFAULT_REPORTS = ROOT / "reports"


def _metrics(y_true, y_prob, threshold: float = 0.5) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": threshold,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_test": int(len(y_true)),
        "positive_rate_test": float(np.mean(y_true)),
    }


def _input_defaults(X: pd.DataFrame) -> dict:
    defaults = {}
    for col in NUMERIC_FEATURES:
        if col in ("avg_monthly_spend", "num_services"):
            continue
        defaults[col] = float(X[col].median())
    for col in CATEGORICAL_FEATURES:
        if col == "tenure_group":
            continue
        defaults[col] = str(X[col].mode().iloc[0])
    return defaults


def _build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def _pipelines(random_state: int) -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            [
                ("preprocess", _build_preprocessor()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocess", _build_preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=80,
                        max_depth=8,
                        min_samples_leaf=5,
                        class_weight="balanced",
                        random_state=random_state,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
    }


def _encoded_feature_names(preprocess: ColumnTransformer) -> list[str]:
    cat = preprocess.named_transformers_["cat"]
    cat_names = list(cat.get_feature_names_out(CATEGORICAL_FEATURES))
    return list(NUMERIC_FEATURES) + cat_names


def _export_logistic(pipeline: Pipeline, defaults: dict, metrics: dict, n_rows: int) -> dict:
    preprocess = pipeline.named_steps["preprocess"]
    model = pipeline.named_steps["model"]
    scaler = preprocess.named_transformers_["num"]
    encoder = preprocess.named_transformers_["cat"]
    names = _encoded_feature_names(preprocess)
    coefs = [float(c) for c in model.coef_[0]]
    weights = sorted(
        [{"name": n, "weight": w} for n, w in zip(names, coefs)],
        key=lambda item: abs(item["weight"]),
        reverse=True,
    )
    categories = {
        feat: [str(v) for v in cats]
        for feat, cats in zip(CATEGORICAL_FEATURES, encoder.categories_)
    }
    return {
        "model": "logistic_regression",
        "threshold": 0.5,
        "intercept": float(model.intercept_[0]),
        "numeric_features": [
            {"name": name, "mean": float(mean), "scale": float(scale)}
            for name, mean, scale in zip(NUMERIC_FEATURES, scaler.mean_, scaler.scale_)
        ],
        "categorical_features": categories,
        "encoded_feature_names": names,
        "coefficients": coefs,
        "feature_weights": weights,
        "input_defaults": defaults,
        "addon_cols": [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ],
        "metrics": metrics,
        "n_rows": n_rows,
        "dataset": {
            "name": DATASET_NAME,
            "source": DATASET_SOURCE,
            "url": DATASET_URL,
            "license": DATASET_LICENSE,
        },
    }


def _save_charts(reports_dir: Path, docs_dir: Path, results: dict, lr_pipeline, X_test, y_test) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    assets = docs_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 5))
    for name, payload in results.items():
        RocCurveDisplay.from_predictions(y_test, payload["y_prob"], name=name.replace("_", " ").title(), ax=ax)
    ax.set_title("ROC curve on the held-out test set")
    fig.tight_layout()
    fig.savefig(reports_dir / "roc_curve.png", dpi=120)
    fig.savefig(assets / "roc_curve.png", dpi=120)
    plt.close(fig)

    cm = results["logistic_regression"]["metrics"]["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(
        confusion_matrix=np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]]),
        display_labels=["Stay", "Churn"],
    ).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Logistic regression confusion matrix")
    fig.tight_layout()
    fig.savefig(reports_dir / "confusion_matrix.png", dpi=120)
    fig.savefig(assets / "confusion_matrix.png", dpi=120)
    plt.close(fig)

    preprocess = lr_pipeline.named_steps["preprocess"]
    coefs = lr_pipeline.named_steps["model"].coef_[0]
    names = _encoded_feature_names(preprocess)
    order = np.argsort(np.abs(coefs))[::-1][:12]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh([names[i] for i in order][::-1], coefs[order][::-1], color="#1f4e79")
    ax.set_xlabel("Coefficient (standardised features)")
    ax.set_title("Top logistic regression weights")
    fig.tight_layout()
    fig.savefig(reports_dir / "feature_weights.png", dpi=120)
    fig.savefig(assets / "feature_weights.png", dpi=120)
    plt.close(fig)


def _write_report(path: Path, metrics: dict, n_rows: int) -> None:
    lr = metrics["logistic_regression"]
    rf = metrics["random_forest"]
    lr_cm = lr["confusion_matrix"]
    rf_cm = rf["confusion_matrix"]
    path.write_text(
        "\n".join(
            [
                "# Churn prediction report",
                "",
                f"Dataset: {DATASET_NAME} ({n_rows} customers after cleaning).",
                f"Source: {DATASET_SOURCE}",
                f"Licence: {DATASET_LICENSE}",
                "",
                "Train/test split: 80/20, stratified, `random_state=42`.",
                "Both models use `class_weight='balanced'`. The decision threshold is 0.5.",
                "",
                "## Test-set metrics",
                "",
                "| Model | ROC-AUC | Precision | Recall | F1 | Accuracy |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
                (
                    f"| Logistic regression | {lr['roc_auc']:.3f} | {lr['precision']:.3f} | "
                    f"{lr['recall']:.3f} | {lr['f1']:.3f} | {lr['accuracy']:.3f} |"
                ),
                (
                    f"| Random forest | {rf['roc_auc']:.3f} | {rf['precision']:.3f} | "
                    f"{rf['recall']:.3f} | {rf['f1']:.3f} | {rf['accuracy']:.3f} |"
                ),
                "",
                "## Confusion matrices (test set)",
                "",
                f"- Logistic regression: TN={lr_cm['tn']}, FP={lr_cm['fp']}, FN={lr_cm['fn']}, TP={lr_cm['tp']}",
                f"- Random forest: TN={rf_cm['tn']}, FP={rf_cm['fp']}, FN={rf_cm['fn']}, TP={rf_cm['tp']}",
                "",
                "## Charts",
                "",
                "![ROC curve](roc_curve.png)",
                "",
                "![Confusion matrix](confusion_matrix.png)",
                "",
                "![Feature weights](feature_weights.png)",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def run(
    data_path: Path | None = None,
    docs_dir: Path | None = None,
    reports_dir: Path | None = None,
    random_state: int = 42,
    test_size: float = 0.2,
    write_charts: bool = True,
) -> dict:
    docs_dir = Path(docs_dir) if docs_dir is not None else DEFAULT_DOCS
    reports_dir = Path(reports_dir) if reports_dir is not None else DEFAULT_REPORTS
    docs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    raw = load_raw(data_path)
    prepared = add_features(clean(raw))
    X, y = model_matrix(prepared)
    if y is None:
        raise ValueError("Training data must include a Churn column")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    results = {}
    fitted = {}
    for name, pipe in _pipelines(random_state).items():
        pipe.fit(X_train, y_train)
        y_prob = pipe.predict_proba(X_test)[:, 1]
        results[name] = {"metrics": _metrics(y_test, y_prob), "y_prob": y_prob}
        fitted[name] = pipe

    metrics = {name: payload["metrics"] for name, payload in results.items()}
    metrics["n_rows"] = int(len(prepared))
    metrics["n_train"] = int(len(X_train))
    metrics["n_test"] = int(len(X_test))
    metrics["churn_rate"] = float(y.mean())

    defaults = _input_defaults(X_train)
    export = _export_logistic(fitted["logistic_regression"], defaults, metrics, int(len(prepared)))

    # A couple of worked examples so the JS demo can be checked against Python.
    demo_rows = X_test.head(3)
    export["examples"] = []
    for _, row in demo_rows.iterrows():
        proba = float(fitted["logistic_regression"].predict_proba(row.to_frame().T)[0, 1])
        export["examples"].append({"input": json.loads(row.to_json()), "probability": proba})

    (reports_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    (docs_dir / "model.json").write_text(json.dumps(export, indent=2) + "\n", encoding="utf-8")
    _write_report(reports_dir / "REPORT.md", metrics, int(len(prepared)))
    if write_charts:
        _save_charts(reports_dir, docs_dir, results, fitted["logistic_regression"], X_test, y_test)

    print(json.dumps({k: v for k, v in metrics.items() if k != "n_rows"}, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train churn models and export the static demo payload")
    parser.add_argument("--data", type=Path, default=None, help="Optional local CSV; downloads the IBM file if omitted")
    parser.add_argument("--docs-dir", type=Path, default=None)
    parser.add_argument("--reports-dir", type=Path, default=None)
    parser.add_argument("--no-charts", action="store_true")
    args = parser.parse_args()
    run(
        data_path=args.data,
        docs_dir=args.docs_dir,
        reports_dir=args.reports_dir,
        write_charts=not args.no_charts,
    )


if __name__ == "__main__":
    main()
