import numpy as np
import pandas as pd
from bayesian_retail.io_utils import *

def test_json_yaml(tmp_path):
    d = {"a": 1}
    write_json(d, tmp_path / "a.json")
    write_yaml(d, tmp_path / "a.yaml")
    assert read_json(tmp_path / "a.json") == d
    assert read_yaml(tmp_path / "a.yaml") == d

def test_dataframe(tmp_path):
    f = pd.DataFrame({"x": [1, 2]})
    write_dataframe(f, tmp_path / "x.csv")
    pd.testing.assert_frame_equal(f, read_dataframe(tmp_path / "x.csv"))

def test_npz(tmp_path):
    write_npz(tmp_path / "x.npz", x=np.array([1, 2]))
    assert np.load(tmp_path / "x.npz")["x"].tolist() == [1, 2]
