import pandas as pd

from ticketintel.config import SAMPLE_COLUMNS, SAMPLE_PATH
from ticketintel.features import FeatureBuilder, request_frame, structured_frame


def test_sample_header_matches_schema():
    header = pd.read_csv(SAMPLE_PATH, nrows=0).columns.tolist()
    assert tuple(header) == SAMPLE_COLUMNS


def test_structured_fields_cover_the_schema():
    frame = request_frame(
        "The checking account was charged twice.",
        "tx",
        "Servicemember",
        "2024-03-12",
    )
    side = structured_frame(frame)
    expected = {
        "text",
        "state_raw",
        "tag_servicemember",
        "tag_older",
        "text_chars",
        "text_words",
        "date_missing",
        "weekday",
        "short_text",
    }
    assert set(side.columns) == expected
    assert side.loc[0, "state_raw"] == "TX"
    assert int(side.loc[0, "tag_servicemember"]) == 1
    assert int(side.loc[0, "date_missing"]) == 0
    assert int(side.loc[0, "weekday"]) == 1  # 2024-03-12 was a Tuesday


def test_unknown_state_maps_to_other():
    builder = FeatureBuilder()
    builder.state_levels = ["CA"]
    frame = request_frame("a new complaint about a fee", "ZZ", None, None)
    matrix = builder._state_matrix(structured_frame(frame))
    columns = builder.state_levels + ["UNK", "OTHER"]
    assert matrix.shape == (1, len(columns))
    assert int(matrix[0, columns.index("OTHER")]) == 1
    assert builder.state_columns()[0] == "state_CA"


def test_request_frame_columns():
    frame = request_frame("hello", None, None, None)
    assert list(frame.columns) == ["narrative", "state", "tags", "date_received"]
    assert frame.loc[0, "narrative"] == "hello"
