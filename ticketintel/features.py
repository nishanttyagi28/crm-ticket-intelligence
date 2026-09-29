"""Text plus structured features for complaint routing.

Company, issue, and sub-product are left out on purpose. They nearly determine
the product line, so they would turn the task into a lookup.
"""

from __future__ import annotations

import math
import re

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler

from ticketintel.config import SHORT_TEXT_CHARS, STATE_MIN_COUNT, TEXT_COLUMN

WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")
WORD_RE = re.compile(r"\S+")


def _clean_text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return " ".join(str(value).split())


def _missing_state(value: object) -> bool:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return True
    text = str(value).strip()
    return text == "" or text.upper() in {"NA", "NAN", "NONE", "NULL"}


def _tag_text(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value)


def parse_received_date(value: object) -> pd.Timestamp | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    if text == "" or text.upper() in {"NA", "NAN", "NONE", "NULL"}:
        return None
    parsed = pd.to_datetime(text, errors="coerce", utc=False)
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed).tz_localize(None)


def structured_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Row-level fields that do not require a fitted vocabulary."""
    text = (
        df[TEXT_COLUMN].map(_clean_text) if TEXT_COLUMN in df.columns else pd.Series([""] * len(df))
    )
    tags = df["tags"].map(_tag_text) if "tags" in df.columns else pd.Series([""] * len(df))
    dates = (
        df["date_received"].map(parse_received_date)
        if "date_received" in df.columns
        else pd.Series([None] * len(df))
    )
    states = df["state"] if "state" in df.columns else pd.Series([None] * len(df))

    words = text.map(lambda item: len(WORD_RE.findall(item)))
    chars = text.map(len)
    date_missing = dates.map(lambda item: 1 if item is None else 0).astype(int)
    weekday = []
    for item, missing in zip(dates, date_missing, strict=True):
        weekday.append(-1 if missing else int(item.dayofweek))

    out = pd.DataFrame(
        {
            "text": text.to_numpy(),
            "state_raw": states.map(
                lambda item: "UNK" if _missing_state(item) else str(item).strip().upper()
            ).to_numpy(),
            "tag_servicemember": tags.map(lambda item: int("Servicemember" in item)).to_numpy(),
            "tag_older": tags.map(lambda item: int("Older American" in item)).to_numpy(),
            "text_chars": chars.to_numpy(dtype=float),
            "text_words": words.to_numpy(dtype=float),
            "date_missing": date_missing.to_numpy(),
            "weekday": weekday,
            "short_text": (chars < SHORT_TEXT_CHARS).astype(int).to_numpy(),
        },
        index=df.index,
    )
    return out


class FeatureBuilder:
    """TF-IDF narrative plus state, weekday, tags, and length."""

    def __init__(self) -> None:
        self.tfidf = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            min_df=4,
            max_features=5000,
            sublinear_tf=True,
            dtype=np.float32,
        )
        self.svd = TruncatedSVD(n_components=80, algorithm="randomized", n_iter=5, random_state=42)
        self.scaler = MaxAbsScaler()
        self.state_levels: list[str] = []
        self.numeric_columns = (
            "text_chars",
            "text_words",
            "date_missing",
            "tag_servicemember",
            "tag_older",
        )

    def fit(self, df: pd.DataFrame) -> FeatureBuilder:
        frame = structured_frame(df)
        self.tfidf.fit(frame["text"])
        text_matrix = self.tfidf.transform(frame["text"])
        self.svd.fit(text_matrix)
        counts = frame["state_raw"].value_counts()
        self.state_levels = sorted(
            state
            for state, count in counts.items()
            if state != "UNK" and int(count) >= STATE_MIN_COUNT
        )
        self.scaler.fit(self._numeric_block(frame))
        return self

    def state_columns(self) -> list[str]:
        return [f"state_{state}" for state in self.state_levels] + ["state_UNK", "state_OTHER"]

    def weekday_columns(self) -> list[str]:
        return [f"weekday_{name}" for name in WEEKDAYS]

    def transform(self, df: pd.DataFrame) -> sparse.csr_matrix:
        frame = structured_frame(df)
        text_matrix = self.tfidf.transform(frame["text"])
        state_matrix = self._state_matrix(frame)
        weekday_matrix = self._weekday_matrix(frame)
        numeric = sparse.csr_matrix(self.scaler.transform(self._numeric_block(frame)))
        return sparse.hstack(
            [text_matrix, state_matrix, weekday_matrix, numeric], format="csr", dtype=np.float32
        )

    def transform_tree(self, df: pd.DataFrame) -> np.ndarray:
        """Dense view: 80 SVD components of the narrative, plus the structured block."""
        frame = structured_frame(df)
        reduced = self.svd.transform(self.tfidf.transform(frame["text"]))
        state_matrix = self._state_matrix(frame).toarray()
        weekday_matrix = self._weekday_matrix(frame).toarray()
        numeric = self.scaler.transform(self._numeric_block(frame))
        return np.hstack([reduced, state_matrix, weekday_matrix, numeric]).astype(np.float32)

    def _numeric_block(self, frame: pd.DataFrame) -> np.ndarray:
        return frame.loc[:, list(self.numeric_columns)].to_numpy(dtype=float)

    def _state_code(self, raw: str) -> str:
        if raw == "UNK":
            return "UNK"
        if raw in self.state_levels:
            return raw
        return "OTHER"

    def _state_matrix(self, frame: pd.DataFrame) -> sparse.csr_matrix:
        columns = self.state_levels + ["UNK", "OTHER"]
        index = {name: pos for pos, name in enumerate(columns)}
        rows = np.arange(len(frame))
        cols = [index[self._state_code(value)] for value in frame["state_raw"]]
        data = np.ones(len(frame), dtype=np.float32)
        return sparse.csr_matrix((data, (rows, cols)), shape=(len(frame), len(columns)))

    def _weekday_matrix(self, frame: pd.DataFrame) -> sparse.csr_matrix:
        rows = []
        cols = []
        for row, weekday in enumerate(frame["weekday"].tolist()):
            if weekday < 0:
                continue
            rows.append(row)
            cols.append(int(weekday))
        data = np.ones(len(rows), dtype=np.float32)
        return sparse.csr_matrix((data, (rows, cols)), shape=(len(frame), 7))


def request_frame(
    text: str, state: str | None, tags: str | None, date_received: str | None
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "narrative": text,
                "state": state or "",
                "tags": tags or "",
                "date_received": date_received or "",
            }
        ]
    )
