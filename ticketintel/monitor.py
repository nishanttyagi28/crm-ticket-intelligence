"""Small batch monitor: class mix versus the training prior, plus missing text."""

from __future__ import annotations

from collections import Counter

from ticketintel.config import MONITOR_TVD_FLAG


def label_shares(labels: list[str], classes: list[str]) -> dict[str, float]:
    counts = Counter(labels)
    total = max(len(labels), 1)
    return {name: counts.get(name, 0) / total for name in classes}


def total_variation(
    reference: dict[str, float], batch: dict[str, float], classes: list[str]
) -> float:
    distance = sum(abs(reference.get(name, 0.0) - batch.get(name, 0.0)) for name in classes)
    return 0.5 * distance


def missing_text_rate(texts: list[str]) -> float:
    if not texts:
        return 0.0
    missing = sum(1 for text in texts if text is None or str(text).strip() == "")
    return missing / len(texts)


def monitor_batch(
    reference_labels: list[str],
    batch_labels: list[str],
    batch_texts: list[str],
    classes: list[str] | None = None,
) -> dict:
    """Compare a scored batch to the training label mix.

    Total variation is 0 when the mixes match and 1 when they are disjoint.
    A flag at 0.15 is a prompt to look, not an automatic rollback.
    """
    ordered = classes or sorted(set(reference_labels) | set(batch_labels))
    reference = label_shares(list(reference_labels), ordered)
    observed = label_shares(list(batch_labels), ordered)
    distance = float(total_variation(reference, observed, ordered))
    return {
        "rows": len(batch_labels),
        "missing_text_rate": float(missing_text_rate(list(batch_texts))),
        "reference_share": {key: round(value, 6) for key, value in reference.items()},
        "batch_share": {key: round(value, 6) for key, value in observed.items()},
        "total_variation": round(distance, 6),
        "flag": bool(distance > MONITOR_TVD_FLAG),
        "flag_threshold": MONITOR_TVD_FLAG,
        "note": (
            "Batch label mix is the model's predicted products on the March holdout, "
            "compared with the January-February training labels."
        ),
    }


def duplicate_rate(values: list[str]) -> float:
    if not values:
        return 0.0
    unique = len(set(values))
    return 1.0 - (unique / len(values))


def class_balance(labels: list[str]) -> dict[str, dict[str, float]]:
    counts = Counter(labels)
    total = max(len(labels), 1)
    ordered = sorted(counts, key=lambda name: (-counts[name], name))
    return {
        name: {"count": int(counts[name]), "share": round(counts[name] / total, 6)}
        for name in ordered
    }
