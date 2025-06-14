import pandas as pd
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scaler import DynamicScaler


def test_reason_present(monkeypatch):
    def fake_fit(self, X, y, kind):
        return None

    def fake_imp(self, model, X):
        return 0.5

    monkeypatch.setattr(DynamicScaler, "_fit_model", fake_fit)
    monkeypatch.setattr(DynamicScaler, "_feature_importance", fake_imp)
    df = pd.read_csv("data/case_data_science_credit.csv", sep=";")
    df_num = df.select_dtypes("number").drop(columns=["client_id", "target"])
    y = df["target"]
    ds = DynamicScaler(strategy="auto", importance_metric="gain").fit(df_num, y)
    rpt = ds.report_as_df()
    assert rpt["reason"].notna().all()
    assert rpt.loc["saldo_rotativo_total", "reason"].startswith("stats")

