import pandas as pd

from bayesian_retail.data.hierarchy import (
    add_hierarchy_indices,
    build_hierarchy_table,
    hierarchy_counts,
)


def test_hierarchy_encoding():
    frame = pd.DataFrame({
        "series_id": ["s1", "s1", "s2", "s2"],
        "item_id": ["i1", "i1", "i2", "i2"],
        "department_id": ["d1", "d1", "d1", "d1"],
        "category_id": ["c1", "c1", "c2", "c2"],
        "store_id": ["st1", "st1", "st1", "st1"],
        "region_id": ["r1", "r1", "r1", "r1"],
        "dataset": ["x", "x", "x", "x"],
    })
    encoded, maps = add_hierarchy_indices(frame)
    hierarchy = build_hierarchy_table(encoded)
    counts = hierarchy_counts(encoded)

    assert len(hierarchy) == 2
    assert counts["series"] == 2
    assert "category_store" in maps
