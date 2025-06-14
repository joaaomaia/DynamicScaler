import os
import sys
import numpy as np
import pandas as pd
import joblib
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scaler import DynamicScaler


def test_minmax_requires_cv_gain(monkeypatch):
    df = pd.DataFrame({"a": np.linspace(0, 10, 200)})
    y = (df["a"] > 5).astype(int)

    def fake_fit(self, X, y_):
        return None

    def fake_imp(self, model, X):
        return 0.8 if X.max() <= 1 else 0.6

    def fake_kurt(arr, **_):
        return 0 if arr.max() <= 1 else 10

    monkeypatch.setattr(DynamicScaler, "_fit_xgb", fake_fit)
    monkeypatch.setattr(DynamicScaler, "_feature_importance", fake_imp)
    monkeypatch.setattr("scaler.kurtosis", fake_kurt)
    scaler = DynamicScaler(
        ignore_scalers=["PowerTransformer", "QuantileTransformer", "RobustScaler"],
        random_state=0,
        scoring=lambda _, arr: arr.std(),
    )
    scaler.fit(df, y)
    assert scaler.report_["a"]["chosen_scaler"] == "MinMaxScaler"


def test_cv_xgboost_enabled_by_flag(monkeypatch):
    df = pd.DataFrame({"a": np.random.lognormal(size=200)})
    y = (df["a"] > 1).astype(int)

    def fake_fit(self, X, y_):
        return None

    def fake_imp(self, model, X):
        return 0.8 if X.max() > 1 else 0.805

    monkeypatch.setattr(DynamicScaler, "_fit_xgb", fake_fit)
    monkeypatch.setattr(DynamicScaler, "_feature_importance", fake_imp)
    monkeypatch.setattr("scaler.kurtosis", lambda arr, **_: 0)
    scaler = DynamicScaler(
        extra_validation=True,
        cv_gain_thr=0.01,
        ignore_scalers=["RobustScaler", "MinMaxScaler"],
        random_state=0,
    )
    scaler.fit(df, y)
    assert scaler.report_["a"]["chosen_scaler"] == "None"


def test_ignore_scalers_skips_minmax():
    df = pd.DataFrame({"a": np.linspace(1.0, 1.001, 100)})
    scaler = DynamicScaler(ignore_scalers=["MinMaxScaler"], random_state=0)
    scaler.fit(df)
    tried = scaler.report_["a"]["candidates_tried"]
    assert "MinMaxScaler" not in tried
    assert tried[-1] == "QuantileTransformer"


def test_serialisation_only_selected(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [1, 1, 1, 1]})
    path = tmp_path / "scalers.pkl"
    scaler = DynamicScaler(serialize=True, save_path=path, random_state=0)
    scaler.fit(df)
    data = joblib.load(path)
    assert "b" not in data["scalers"]

