TARGET_COL="credit_card_default"

INDEX_COL="customer_id"
DROP_COLS=["name"]
NOMINAL_COLS=["gender"]
MAP_COLS=["owns_car", "owns_house"]
MAP_DICT={"Y": 1.0, "N": 0.0}
FILTER_VALUES={"gender": ["XNA"]}

NUMERIC_TYPE="float32"

ORDINAL_COLS=["occupation_type"]
ORDINAL_ORDER=[
    [
        "Unknown",
        "Low-skill Laborers",
        "Cleaning staff",
        "Cooking staff",
        "Waiters/barmen staff",
        "Laborers",
        "Drivers",
        "Security staff",
        "Sales staff",
        "Private service staff",
        "Medicine staff",
        "Core staff",
        "Accountants",
        "Secretaries",
        "HR staff",
        "Realty agents",
        "IT staff",
        "High skill tech staff",
        "Managers"
    ]
]

TEST_SIZE=0.2
RANDOM_STATE=2026