import json

from ticketintel.config import METRICS_PATH, QUALITY_PATH

REQUIRED = ("accuracy", "precision_macro", "recall_macro", "f1_macro")


def test_metrics_file_has_holdout_scores():
    assert METRICS_PATH.exists()
    report = json.loads(METRICS_PATH.read_text())
    assert report["split"]["method"] == "chronological"
    assert report["split"]["train_rows"] > 0
    assert report["split"]["test_rows"] > 0
    names = set(report["models"])
    assert names >= {"logistic_regression", "linear_svc", "complement_nb", "random_forest"}
    for name, block in report["models"].items():
        for key in REQUIRED:
            value = block[key]
            assert 0.0 <= value <= 1.0, name
        auc = block["roc_auc_ovr_macro"]
        assert auc is None or 0.0 <= auc <= 1.0
    winner = report["winner"]
    best = max(report["models"], key=lambda name: report["models"][name]["f1_macro"])
    assert winner == best


def test_data_quality_report():
    report = json.loads(QUALITY_PATH.read_text())
    assert 0.0 <= report["missing_text_rate"] <= 1.0
    assert 0.0 <= report["duplicate_narrative_rate"] <= 1.0
    assert 0.0 <= report["duplicate_complaint_id_rate"] <= 1.0
    assert report["class_balance"]
    assert report["weak_slice"]["n"] >= 200
    assert report["weak_slice"]["accuracy"] <= report["weak_slice"]["overall_accuracy"]
