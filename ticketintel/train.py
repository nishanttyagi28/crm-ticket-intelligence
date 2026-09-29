"""Fit routing models and write metrics from the chronological holdout."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

from ticketintel.config import (
    LABEL_COLUMN,
    LABEL_DEFINITION,
    METRICS_PATH,
    MODEL_PATH,
    MONITOR_PATH,
    QUALITY_PATH,
    SAMPLE_PATH,
    TEST_END,
    TRAIN_END,
    TRAIN_START,
    UNIVERSE_STATS_PATH,
    WEAK_SLICE_MIN_ROWS,
)
from ticketintel.features import FeatureBuilder, structured_frame
from ticketintel.monitor import class_balance, duplicate_rate, monitor_batch

RANDOM_STATE = 42


def load_sample() -> pd.DataFrame:
    frame = pd.read_csv(SAMPLE_PATH, dtype=str, keep_default_na=False)
    frame["date_received"] = frame["date_received"].str.slice(0, 10)
    return frame.sort_values(["date_received", "complaint_id"]).reset_index(drop=True)


def dedupe_narratives(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep the earliest complaint when the same narrative was filed more than once.

    The extract is already sorted by date and complaint id. Dropping later copies
    stops a repeated write-up from landing on both sides of the holdout.
    """
    return frame.drop_duplicates(subset=["narrative"], keep="first").reset_index(drop=True)


def split_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = frame.loc[frame["date_received"] < TRAIN_END].copy()
    test = frame.loc[
        (frame["date_received"] >= TRAIN_END) & (frame["date_received"] < TEST_END)
    ].copy()
    return train, test


def _scores(estimator, matrix) -> np.ndarray | None:
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(matrix)
    if hasattr(estimator, "decision_function"):
        return estimator.decision_function(matrix)
    return None


def _metric_block(y_true, y_pred, y_score, classes: list[str]) -> dict:
    block = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "roc_auc_ovr_macro": None,
    }
    if y_score is not None and len(classes) > 1:
        try:
            block["roc_auc_ovr_macro"] = float(
                roc_auc_score(
                    y_true,
                    y_score,
                    multi_class="ovr",
                    average="macro",
                    labels=classes,
                )
            )
        except ValueError:
            block["roc_auc_ovr_macro"] = None
    return {
        key: (None if value is None else round(float(value), 6)) for key, value in block.items()
    }


def _per_class_f1(y_true, y_pred, classes: list[str]) -> dict[str, float]:
    scores = f1_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    return {name: round(float(score), 6) for name, score in zip(classes, scores, strict=True)}


def _slice_accuracy(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> dict | None:
    count = int(mask.sum())
    if count < WEAK_SLICE_MIN_ROWS:
        return None
    return {
        "n": count,
        "accuracy": round(float(accuracy_score(y_true[mask], y_pred[mask])), 6),
    }


def weak_slice(test: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, overall: float) -> dict:
    """Pick the predefined slice with the lowest accuracy and enough rows."""
    side = structured_frame(test)
    candidates: list[dict] = []

    short = _slice_accuracy(y_true, y_pred, side["short_text"].to_numpy() == 1)
    if short:
        short["name"] = "short_narrative"
        short["definition"] = "Test complaints whose narrative is under 200 characters."
        candidates.append(short)

    service = _slice_accuracy(y_true, y_pred, side["tag_servicemember"].to_numpy() == 1)
    if service:
        service["name"] = "servicemember_tag"
        service["definition"] = "Test complaints tagged Servicemember."
        candidates.append(service)

    older = _slice_accuracy(y_true, y_pred, side["tag_older"].to_numpy() == 1)
    if older:
        older["name"] = "older_american_tag"
        older["definition"] = "Test complaints tagged Older American."
        candidates.append(older)

    missing_state = _slice_accuracy(y_true, y_pred, side["state_raw"].to_numpy() == "UNK")
    if missing_state:
        missing_state["name"] = "missing_state"
        missing_state["definition"] = "Test complaints with no state code."
        candidates.append(missing_state)

    for product in sorted(set(y_true)):
        block = _slice_accuracy(y_true, y_pred, y_true == product)
        if block:
            block["name"] = f"product:{product}"
            block["definition"] = f"Test complaints whose true product is {product}."
            candidates.append(block)

    if not candidates:
        return {"name": None, "definition": "No slice met the row minimum.", "n": 0}

    chosen = min(candidates, key=lambda item: (item["accuracy"], -item["n"], item["name"]))
    chosen["overall_accuracy"] = round(float(overall), 6)
    chosen["accuracy_gap"] = round(float(overall - chosen["accuracy"]), 6)
    chosen["selection_rule"] = (
        f"Lowest accuracy among predefined slices with at least {WEAK_SLICE_MIN_ROWS} test rows."
    )
    chosen["candidates"] = candidates
    return chosen


def build_estimators() -> dict:
    return {
        "logistic_regression": (
            "sparse",
            LogisticRegression(
                solver="saga",
                C=1.0,
                max_iter=200,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            {
                "solver": "saga",
                "C": 1.0,
                "max_iter": 200,
                "class_weight": "balanced",
                "random_state": RANDOM_STATE,
            },
        ),
        "linear_svc": (
            "sparse",
            LinearSVC(C=1.0, class_weight="balanced", dual="auto", random_state=RANDOM_STATE),
            {"C": 1.0, "class_weight": "balanced", "dual": "auto"},
        ),
        "complement_nb": (
            "sparse",
            ComplementNB(alpha=0.5),
            {"alpha": 0.5},
        ),
        "random_forest": (
            "tree",
            RandomForestClassifier(
                n_estimators=200,
                max_depth=18,
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=4,
            ),
            {
                "n_estimators": 200,
                "max_depth": 18,
                "min_samples_leaf": 5,
                "class_weight": "balanced_subsample",
                "features": "80-component truncated SVD of TF-IDF plus the structured block",
            },
        ),
    }


def data_quality(
    extract: pd.DataFrame,
    frame: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    chosen_slice: dict,
) -> dict:
    universe = json.loads(UNIVERSE_STATS_PATH.read_text())
    public_slice = {
        key: chosen_slice[key]
        for key in (
            "name",
            "definition",
            "n",
            "accuracy",
            "overall_accuracy",
            "accuracy_gap",
            "selection_rule",
        )
        if key in chosen_slice
    }
    return {
        "universe_rows": universe["universe_rows"],
        "missing_text_rows": universe["missing_text_rows"],
        "missing_text_rate": universe["missing_text_rate"],
        "missing_text_definition": (
            "Share of the date-and-product universe whose narrative is null or blank, "
            "before the 40-character modeling filter."
        ),
        "modeling_rows": int(len(extract)),
        "rows_after_narrative_dedup": int(len(frame)),
        "modeling_missing_text_rate": 0.0,
        "duplicate_complaint_id_rate": round(duplicate_rate(extract["complaint_id"].tolist()), 6),
        "duplicate_narrative_rate": round(duplicate_rate(extract["narrative"].tolist()), 6),
        "duplicate_narrative_definition": (
            "One minus the number of distinct narratives divided by modeling rows. "
            "Complaint ids in this extract are unique."
        ),
        "class_balance": class_balance(extract[LABEL_COLUMN].tolist()),
        "class_balance_after_dedup": class_balance(frame[LABEL_COLUMN].tolist()),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "submitted_via_in_sample": universe["submitted_via_in_sample"],
        "channel_note": (
            "Every narrative in this window was submitted via Web, so channel is not a feature."
        ),
        "weak_slice": public_slice,
    }


def main() -> None:
    extract = load_sample()
    frame = dedupe_narratives(extract)
    train, test = split_frame(frame)
    if train.empty or test.empty:
        raise SystemExit("Train or test split is empty. Check the sample dates.")

    builder = FeatureBuilder()
    print(f"fitting features on {len(train)} train rows", flush=True)
    builder.fit(train)
    x_train = builder.transform(train)
    x_test = builder.transform(test)
    x_train_tree = builder.transform_tree(train)
    x_test_tree = builder.transform_tree(test)
    y_train = train[LABEL_COLUMN].to_numpy()
    y_test = test[LABEL_COLUMN].to_numpy()
    results = {}
    fitted = {}
    specs = build_estimators()
    for name, (kind, estimator, params) in specs.items():
        print(f"fitting {name}", flush=True)
        matrix_train = x_train_tree if kind == "tree" else x_train
        matrix_test = x_test_tree if kind == "tree" else x_test
        estimator.fit(matrix_train, y_train)
        pred = estimator.predict(matrix_test)
        score = _scores(estimator, matrix_test)
        block = _metric_block(y_test, pred, score, list(estimator.classes_))
        block["params"] = params
        block["per_class_f1"] = _per_class_f1(y_test, pred, list(estimator.classes_))
        results[name] = block
        fitted[name] = (kind, estimator, pred)
        print(f"  f1_macro={block['f1_macro']} accuracy={block['accuracy']}", flush=True)

    winner = max(
        results,
        key=lambda name: (
            results[name]["f1_macro"],
            results[name]["roc_auc_ovr_macro"] or -1,
            results[name]["accuracy"],
        ),
    )
    kind, estimator, pred = fitted[winner]
    slice_report = weak_slice(test, y_test, pred, results[winner]["accuracy"])
    quality = data_quality(extract, frame, train, test, slice_report)
    monitor = monitor_batch(
        reference_labels=y_train.tolist(),
        batch_labels=pred.tolist(),
        batch_texts=test["narrative"].tolist(),
        classes=sorted(set(y_train)),
    )

    metrics = {
        "label": LABEL_COLUMN,
        "label_definition": LABEL_DEFINITION,
        "split": {
            "method": "chronological",
            "train_start": TRAIN_START,
            "train_end_exclusive": TRAIN_END,
            "test_start": TRAIN_END,
            "test_end_exclusive": TEST_END,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "extract_rows": int(len(extract)),
            "rows_after_narrative_dedup": int(len(frame)),
        },
        "features": {
            "text": "TF-IDF word unigrams and bigrams, min_df=4, max_features=5000, sublinear_tf",
            "structured": [
                "state (levels with at least 200 training rows, plus UNK and OTHER)",
                "weekday of date_received",
                "date_missing",
                "tag_servicemember",
                "tag_older",
                "text_chars and text_words, max-abs scaled on the training split",
            ],
            "not_used": [
                "company",
                "sub_product",
                "issue",
                "sub_issue",
                "company_response",
                "timely_response",
                "submitted_via (constant Web in this extract)",
                "month (the test month is entirely after training)",
            ],
        },
        "models": results,
        "winner": winner,
        "winner_rule": (
            "Highest macro-F1 on the March 2024 holdout. "
            "Ties break on macro OVR ROC-AUC, then accuracy."
        ),
        "winner_per_class_f1": results[winner]["per_class_f1"],
    }

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2) + "\n")
    QUALITY_PATH.write_text(json.dumps(quality, indent=2) + "\n")
    MONITOR_PATH.write_text(json.dumps(monitor, indent=2) + "\n")
    joblib.dump(
        {
            "builder": builder,
            "estimator": estimator,
            "model_name": winner,
            "kind": kind,
            "classes": list(estimator.classes_),
        },
        MODEL_PATH,
    )
    print(f"winner={winner}", flush=True)


if __name__ == "__main__":
    main()
