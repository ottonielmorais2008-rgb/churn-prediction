from churn.train import run


def test_train_smoke(tmp_path, sample_df):
    raw = tmp_path / "sample.csv"
    sample_df.to_csv(raw, index=False)
    docs = tmp_path / "docs"
    reports = tmp_path / "reports"
    metrics = run(
        data_path=raw,
        docs_dir=docs,
        reports_dir=reports,
        write_charts=False,
        test_size=0.25,
    )
    assert (docs / "model.json").exists()
    assert (reports / "metrics.json").exists()
    assert (reports / "REPORT.md").exists()
    for name in ("logistic_regression", "random_forest"):
        block = metrics[name]
        for key in ("roc_auc", "precision", "recall", "f1"):
            assert key in block
            assert 0.0 <= block[key] <= 1.0
        cm = block["confusion_matrix"]
        assert cm["tn"] + cm["fp"] + cm["fn"] + cm["tp"] == block["n_test"]
