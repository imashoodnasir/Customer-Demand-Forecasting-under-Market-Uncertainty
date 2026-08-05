import pandas as pd
import pytest

from bayesian_retail.data.schema import UnifiedRetailSchema
from bayesian_retail.exceptions import DataValidationError


def valid_frame():
    return pd.DataFrame({
        "dataset": ["x"],
        "series_id": ["s1"],
        "date": [pd.Timestamp("2024-01-01")],
        "demand": [1],
        "item_id": ["i1"],
        "department_id": ["d1"],
        "category_id": ["c1"],
        "store_id": ["st1"],
        "region_id": ["r1"],
        "price": [1.0],
        "promotion": [0],
        "holiday": [0],
        "snap": [0],
        "transactions": [1.0],
        "oil_price": [70.0],
    })


def test_valid_schema():
    UnifiedRetailSchema().validate(valid_frame())


def test_missing_column():
    frame = valid_frame().drop(columns=["demand"])
    with pytest.raises(DataValidationError):
        UnifiedRetailSchema().validate(frame)


def test_duplicate_series_date():
    frame = pd.concat([valid_frame(), valid_frame()], ignore_index=True)
    with pytest.raises(DataValidationError):
        UnifiedRetailSchema().validate(frame)
