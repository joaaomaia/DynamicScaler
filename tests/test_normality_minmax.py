import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scaler import DynamicScaler


def test_standard_on_normal_data():
    np.random.seed(0)
    df = pd.DataFrame({"a": np.random.normal(0, 1, 500)})
    scaler = DynamicScaler(random_state=0)
    scaler.fit(df)
    assert scaler.report_["a"]["chosen_scaler"] == "RobustScaler"


def test_skip_standard_on_skewed():
    np.random.seed(0)
    df = pd.DataFrame({"a": np.random.lognormal(size=300)})
    scaler = DynamicScaler(random_state=0)
    scaler.fit(df)
    assert "StandardScaler" not in scaler.report_["a"]["candidates_tried"]


def test_minmax_as_last_resort():
    df = pd.DataFrame({"a": np.linspace(1.0, 1.001, 100)})
    scaler = DynamicScaler(
        min_post_std=2.0,
        min_post_iqr=2.0,
        scoring=lambda _y, arr: arr.std(),
        random_state=0,
    )
    scaler.fit(df)
    report = scaler.report_["a"]
    if report["chosen_scaler"] != "MinMaxScaler":
        assert "MinMaxScaler" in report["candidates_tried"]
    else:
        assert report["chosen_scaler"] == "MinMaxScaler"
