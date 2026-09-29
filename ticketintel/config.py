"""Shared paths and the documented sample definition."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REPORT_DIR = ROOT / "reports"
MODEL_DIR = ROOT / "models"

SAMPLE_PATH = DATA_DIR / "sample_complaints.csv.gz"
UNIVERSE_STATS_PATH = DATA_DIR / "universe_stats.json"
METRICS_PATH = REPORT_DIR / "metrics.json"
QUALITY_PATH = REPORT_DIR / "data_quality.json"
MONITOR_PATH = REPORT_DIR / "monitor.json"
MODEL_PATH = MODEL_DIR / "winner.joblib"

# Chronological holdout. Dates are inclusive of the start and exclusive of the end.
TRAIN_START = "2024-01-01"
TRAIN_END = "2024-03-01"
TEST_END = "2024-04-01"

MIN_NARRATIVE_CHARS = 40

# Credit-reporting rows dominate the quarter and are a different dispute workflow.
# Debt or credit management has too few rows for a stable class.
EXCLUDED_PRODUCT_PREFIXES = ("Credit reporting",)
EXCLUDED_PRODUCTS = ("Debt or credit management",)

LABEL_COLUMN = "product"
TEXT_COLUMN = "narrative"
ID_COLUMN = "complaint_id"

SAMPLE_COLUMNS = (
    "complaint_id",
    "date_received",
    "product",
    "narrative",
    "state",
    "tags",
    "submitted_via",
    "zip_code",
)

STATE_MIN_COUNT = 200
SHORT_TEXT_CHARS = 200
WEAK_SLICE_MIN_ROWS = 200
MONITOR_TVD_FLAG = 0.15

SOURCE_URL = (
    "https://huggingface.co/datasets/vivekkopthsd/cfpb-consumer-complaints"
    "/resolve/main/complaints.parquet"
)
SOURCE_PAGE = "https://huggingface.co/datasets/vivekkopthsd/cfpb-consumer-complaints"
OFFICIAL_PAGE = "https://www.consumerfinance.gov/data-research/consumer-complaints/"
SOURCE_AS_OF = "2026-08-11"

LABEL_DEFINITION = (
    "The label is the Product value the consumer filed with the CFPB. "
    "It is the product line a helpdesk would route the complaint to. "
    "The model does not see product, sub-product, issue, company, or the company's response."
)
