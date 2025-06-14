import pandas as pd
import numpy as np
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scaler import DynamicScaler


def test_passthrough_keeps_values():
    X = pd.DataFrame({"x": np.random.rand(200)})
    ds = DynamicScaler(ignore_scalers=["PowerTransformer"]).fit(X)
    assert ds.report_["x"]["chosen_scaler"] == "None"
    X_tr = ds.transform(X, return_df=True)
    assert np.allclose(X["x"].values, X_tr["x"].values)
