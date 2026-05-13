from pathlib import Path

RANDOM_STATE = 42
TEST_SIZE = 0.20

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "data.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
MODELS_DIR = OUTPUT_DIR / "models"

TARGET_COL = "Target"
DROPOUT_POSITIVE_LABEL = "Dropout"

# Integer-coded nominal variables that should be one-hot encoded, not treated as continuous.
CATEGORICAL_FEATURES = [
    "Marital status",
    "Application mode",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nacionality",  # Original dataset uses this spelling.
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
]

# Already coded as 0/1 in the raw dataset. We explicitly validate and cast these to integers.
BINARY_FEATURES = [
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]

ORDINAL_NUMERIC_FEATURES = [
    "Application order",
]

CONTINUOUS_FEATURES = [
    "Previous qualification (grade)",
    "Admission grade",
    "Age at enrollment",
    "Curricular units 1st sem (credited)",
    "Curricular units 1st sem (enrolled)",
    "Curricular units 1st sem (evaluations)",
    "Curricular units 1st sem (approved)",
    "Curricular units 1st sem (grade)",
    "Curricular units 1st sem (without evaluations)",
    "Curricular units 2nd sem (credited)",
    "Curricular units 2nd sem (enrolled)",
    "Curricular units 2nd sem (evaluations)",
    "Curricular units 2nd sem (approved)",
    "Curricular units 2nd sem (grade)",
    "Curricular units 2nd sem (without evaluations)",
    "Unemployment rate",
    "Inflation rate",
    "GDP",
]

ENGINEERED_NUMERIC_FEATURES = [
    "Total curricular units enrolled",
    "Total curricular units approved",
    "Overall approval rate",
    "1st sem approval rate",
    "2nd sem approval rate",
    "Grade change 2nd minus 1st sem",
    "Admission minus previous qualification grade",
]

ENGINEERED_BINARY_FEATURES = [
    "No 1st sem approvals",
    "No 2nd sem approvals",
]

ALL_FEATURES = (
    CATEGORICAL_FEATURES
    + BINARY_FEATURES
    + ORDINAL_NUMERIC_FEATURES
    + CONTINUOUS_FEATURES
    + ENGINEERED_NUMERIC_FEATURES
    + ENGINEERED_BINARY_FEATURES
)
