"""Build the committed complaint sample from the CFPB parquet mirror.

The live CFPB download stopped including narratives in September 2026.
This script reads the public parquet mirror dated 2026-08-11, which still
has the consumer complaint narrative, and writes a filtered extract.

Usage:
    CFPB_PARQUET=/path/to/complaints.parquet python scripts/build_sample.py
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import duckdb

from ticketintel.config import (
    EXCLUDED_PRODUCT_PREFIXES,
    EXCLUDED_PRODUCTS,
    MIN_NARRATIVE_CHARS,
    SAMPLE_COLUMNS,
    SAMPLE_PATH,
    SOURCE_AS_OF,
    SOURCE_PAGE,
    SOURCE_URL,
    TEST_END,
    TRAIN_END,
    TRAIN_START,
    UNIVERSE_STATS_PATH,
)

DEFAULT_PARQUET = Path("/tmp/cfpb/complaints.parquet")


def main() -> None:
    parquet = Path(os.environ.get("CFPB_PARQUET", DEFAULT_PARQUET))
    if not parquet.exists():
        raise SystemExit(f"Parquet not found: {parquet}")

    prefix = EXCLUDED_PRODUCT_PREFIXES[0]
    excluded_product = EXCLUDED_PRODUCTS[0]
    con = duckdb.connect()
    source_rows = con.execute("SELECT count(*) FROM read_parquet(?)", [str(parquet)]).fetchone()[0]

    universe_sql = """
        SELECT
            complaint_id,
            CAST(date_received AS DATE) AS date_received,
            product,
            narrative,
            state,
            tags,
            submitted_via,
            zip_code,
            length(trim(coalesce(narrative, ''))) AS narrative_chars
        FROM read_parquet(?)
        WHERE date_received >= CAST(? AS TIMESTAMP)
          AND date_received < CAST(? AS TIMESTAMP)
          AND product NOT LIKE ? || '%'
          AND product <> ?
    """
    frame = con.execute(
        universe_sql,
        [str(parquet), TRAIN_START, TEST_END, prefix, excluded_product],
    ).fetchdf()

    blank = frame["narrative_chars"] == 0
    short = (frame["narrative_chars"] > 0) & (frame["narrative_chars"] < MIN_NARRATIVE_CHARS)
    kept = frame["narrative_chars"] >= MIN_NARRATIVE_CHARS
    sample = frame.loc[kept, list(SAMPLE_COLUMNS)].copy()
    for column in SAMPLE_COLUMNS:
        sample[column] = sample[column].fillna("").astype(str)
    sample["date_received"] = sample["date_received"].str.slice(0, 10)
    sample = sample.sort_values(["date_received", "complaint_id"]).reset_index(drop=True)

    credit_rows = con.execute(
        """
        SELECT count(*)
        FROM read_parquet(?)
        WHERE date_received >= CAST(? AS TIMESTAMP)
          AND date_received < CAST(? AS TIMESTAMP)
          AND length(trim(coalesce(narrative, ''))) >= ?
          AND product LIKE ? || '%'
        """,
        [str(parquet), TRAIN_START, TEST_END, MIN_NARRATIVE_CHARS, prefix],
    ).fetchone()[0]
    debt_mgmt_rows = con.execute(
        """
        SELECT count(*)
        FROM read_parquet(?)
        WHERE date_received >= CAST(? AS TIMESTAMP)
          AND date_received < CAST(? AS TIMESTAMP)
          AND length(trim(coalesce(narrative, ''))) >= ?
          AND product = ?
        """,
        [str(parquet), TRAIN_START, TEST_END, MIN_NARRATIVE_CHARS, excluded_product],
    ).fetchone()[0]

    dates = sample["date_received"]
    train_rows = int((dates < TRAIN_END).sum())
    test_rows = int((dates >= TRAIN_END).sum())

    SAMPLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(SAMPLE_PATH, index=False, compression="gzip")
    digest = hashlib.sha256(SAMPLE_PATH.read_bytes()).hexdigest()

    submitted = frame.loc[kept, "submitted_via"].fillna("").value_counts().to_dict()
    stats = {
        "source_url": SOURCE_URL,
        "source_page": SOURCE_PAGE,
        "source_as_of": SOURCE_AS_OF,
        "source_rows": int(source_rows),
        "sha256": digest,
        "filters": {
            "date_received_start": TRAIN_START,
            "date_received_end_exclusive": TEST_END,
            "min_narrative_chars": MIN_NARRATIVE_CHARS,
            "excluded_product_prefix": prefix,
            "excluded_products": [excluded_product],
            "train_end_exclusive": TRAIN_END,
        },
        "universe_rows": int(len(frame)),
        "missing_text_rows": int(blank.sum()),
        "missing_text_rate": round(float(blank.mean()) if len(frame) else 0.0, 6),
        "short_text_rows": int(short.sum()),
        "modeling_rows": int(len(sample)),
        "train_rows": train_rows,
        "test_rows": test_rows,
        "duplicate_complaint_id_rate": round(
            1.0 - (sample["complaint_id"].nunique() / len(sample)), 6
        ),
        "duplicate_narrative_rate": round(1.0 - (sample["narrative"].nunique() / len(sample)), 6),
        "excluded_credit_reporting_rows_with_text": int(credit_rows),
        "excluded_debt_or_credit_management_rows_with_text": int(debt_mgmt_rows),
        "submitted_via_in_sample": {str(key): int(value) for key, value in submitted.items()},
        "class_counts": {
            str(key): int(value)
            for key, value in sample["product"].value_counts().sort_values(ascending=False).items()
        },
    }
    UNIVERSE_STATS_PATH.write_text(json.dumps(stats, indent=2) + "\n")
    print(
        json.dumps(
            {k: stats[k] for k in ("modeling_rows", "train_rows", "test_rows", "sha256")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
