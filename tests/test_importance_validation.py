import os, sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scaler import DynamicScaler


def _base_df():
    df = pd.DataFrame({"a": np.linspace(0, 10, 200)})
    y = (df["a"] > 5).astype(int)
    return df, y


def _patch_common(monkeypatch, imp_values):
    def fake_fit(self, X, y):
        return None

    calls = {"n": 0}

    def fake_imp(self, model, X):
        calls["n"] += 1
        return imp_values[calls["n"] - 1]

    def fake_kurt(arr, **_):
        return 10 if arr.max() > 1 else 0

    monkeypatch.setattr(DynamicScaler, "_fit_xgb", fake_fit)
    monkeypatch.setattr(DynamicScaler, "_feature_importance", fake_imp)
    monkeypatch.setattr("scaler.kurtosis", fake_kurt)


def test_importance_gain_shap(monkeypatch):
    df, y = _base_df()
    _patch_common(monkeypatch, [1.0, 1.2])
    ds = DynamicScaler(
        ignore_scalers=["PowerTransformer", "QuantileTransformer", "RobustScaler", "StandardScaler"],
        scoring=lambda _, arr: arr.std(),
        allow_minmax=True,
    )
    ds.fit(df, y)
    assert ds.report_["a"]["chosen_scaler"] == "MinMaxScaler"


def test_importance_no_gain(monkeypatch):
    df, y = _base_df()
    _patch_common(monkeypatch, [1.0, 1.05])
    ds = DynamicScaler(
        ignore_scalers=["PowerTransformer", "QuantileTransformer", "RobustScaler", "StandardScaler"],
        scoring=lambda _, arr: arr.std(),
        allow_minmax=True,
    )
    ds.fit(df, y)
    assert ds.report_["a"]["chosen_scaler"] == "None"


def test_custom_metric_callable(monkeypatch):
    df, y = _base_df()

    def fake_fit(self, X, y):
        return None

    monkeypatch.setattr(DynamicScaler, "_fit_xgb", fake_fit)
    def fake_kurt(arr, **_):
        return 10 if arr.max() > 1 else 0

    monkeypatch.setattr("scaler.kurtosis", fake_kurt)

    ds = DynamicScaler(
        ignore_scalers=["PowerTransformer", "QuantileTransformer", "RobustScaler", "StandardScaler"],
        scoring=lambda _, arr: arr.std(),
        allow_minmax=True,
        importance_metric=lambda m, X: -np.std(X),
    )
    ds.fit(df, y)
    assert ds.report_["a"]["chosen_scaler"] == "MinMaxScaler"
