import pandas as pd
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scaler import DynamicScaler


def test_reason_present():
    df = pd.read_csv("data/case_data_science_credit.csv", sep=";")
    df_num = df.select_dtypes("number").drop(columns=["client_id", "target"])
    y = df["target"]
    ds = DynamicScaler(strategy="auto", importance_metric="gain").fit(df_num, y)
    rpt = ds.report_as_df()
    assert rpt["reason"].notna().all()
    assert rpt.loc["saldo_rotativo_total", "reason"].startswith("stats")

