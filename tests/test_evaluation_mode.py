import os, sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scaler import DynamicScaler


def _patch_kurt(monkeypatch):
    monkeypatch.setattr("scaler.kurtosis", lambda arr, **_: 10 if arr.max() > 1 else 0)


def test_linear_mode_accepts_coefficient_gain(monkeypatch):
    _patch_kurt(monkeypatch)
    df = pd.DataFrame({"a": np.linspace(0, 1000, 200)})
    y = (df["a"] > 500).astype(int)
    ds = DynamicScaler(
        ignore_scalers=["PowerTransformer", "QuantileTransformer", "RobustScaler", "StandardScaler"],
        allow_minmax=True,
        importance_metric="coef",
        evaluation_mode="linear",
    )
    ds.fit(df, y)
    assert ds.report_["a"]["chosen_scaler"] == "MinMaxScaler"


def test_nonlinear_mode_unaffected_by_linear_gain(monkeypatch):
    _patch_kurt(monkeypatch)

    def fake_fit(self, X, y, kind):
        return None

    calls = {"n": 0}

    def fake_imp(self, model, X):
        vals = [1.0, 1.0]  # baseline, candidate
        val = vals[calls["n"]]
        calls["n"] += 1
        return val

    monkeypatch.setattr(DynamicScaler, "_fit_model", fake_fit)
    monkeypatch.setattr(DynamicScaler, "_feature_importance", fake_imp)

    df = pd.DataFrame({"a": np.linspace(0, 1000, 200)})
    y = (df["a"] > 500).astype(int)
    ds = DynamicScaler(
        ignore_scalers=["PowerTransformer", "QuantileTransformer", "RobustScaler", "StandardScaler"],
        allow_minmax=True,
        importance_metric="coef",
        evaluation_mode="nonlinear",
    )
    ds.fit(df, y)
    assert ds.report_["a"]["chosen_scaler"] == "None"


def test_both_mode_average_gain(monkeypatch):
    _patch_kurt(monkeypatch)

    def fake_fit(self, X, y, kind):
        return None

    calls = {"n": 0}

    def fake_imp(self, model, X):
        vals = [1.0, 1.0, 1.0, 1.2]  # logreg base, xgb base, logreg cand, xgb cand
        val = vals[calls["n"]]
        calls["n"] += 1
        return val

    monkeypatch.setattr(DynamicScaler, "_fit_model", fake_fit)
    monkeypatch.setattr(DynamicScaler, "_feature_importance", fake_imp)

    df = pd.DataFrame({"a": np.linspace(0, 1000, 200)})
    y = (df["a"] > 500).astype(int)
    ds = DynamicScaler(
        ignore_scalers=["PowerTransformer", "QuantileTransformer", "RobustScaler", "StandardScaler"],
        allow_minmax=True,
        importance_metric="coef",
        evaluation_mode="both",
    )
    ds.fit(df, y)
    assert ds.report_["a"]["chosen_scaler"] == "MinMaxScaler"
