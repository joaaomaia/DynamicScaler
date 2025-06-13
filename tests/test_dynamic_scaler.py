import os
import sys
import pandas as pd
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scaler import DynamicScaler
from sklearn.exceptions import NotFittedError


def test_transform_preserves_extra_columns():
    df = pd.DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    scaler = DynamicScaler(strategy='standard')
    scaler.fit(df[['a']])
    transformed = scaler.transform(df, return_df=True)
    assert list(transformed.columns) == ['a', 'b']
    pd.testing.assert_series_equal(transformed['b'], df['b'])


def test_inverse_transform_shape():
    df = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [4.0, 5.0, 6.0]})
    scaler = DynamicScaler(strategy='standard')
    scaler.fit(df)
    transformed = scaler.transform(df, return_df=True)
    recovered = scaler.inverse_transform(transformed, return_df=True)
    assert recovered.shape == df.shape


def test_check_is_fitted():
    scaler = DynamicScaler(strategy='standard')
    with pytest.raises(NotFittedError):
        scaler.transform(pd.DataFrame({'a': [1, 2]}))
