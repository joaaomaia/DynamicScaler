import os
import sys
import pandas as pd
import pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scaler import DynamicScaler
from sklearn.exceptions import NotFittedError
import numpy as np


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


def test_ignore_cols_no_scaling():
    df = pd.DataFrame({'a': [1.0, 2.0, 3.0], 'b': [10.0, 20.0, 30.0]})
    scaler = DynamicScaler(strategy='standard', ignore_cols=['b'])
    scaler.fit(df)
    transformed = scaler.transform(df, return_df=True)
    pd.testing.assert_series_equal(transformed['b'], df['b'])
    assert 'b' not in scaler.scalers_


def test_collapse_rejected():
    np.random.seed(42)
    x = np.concatenate([np.random.normal(0, 1, 95), np.random.normal(10, 1, 5)])
    df = pd.DataFrame({'a': x})
    scaler = DynamicScaler(min_post_std=1.5, scoring=lambda _, arr: arr.std(), random_state=42)
    scaler.fit(df)
    assert scaler.report_['a']['chosen_scaler'] == 'RobustScaler'


def test_ignore_scalers():
    df = pd.DataFrame({'a': [1, 2, 3, 4, 5]})
    scaler = DynamicScaler(ignore_scalers=['PowerTransformer'], scoring=lambda _, arr: arr.std())
    scaler.fit(df)
    assert scaler.report_['a']['candidates_tried'][0] == 'QuantileTransformer'


def test_all_rejected():
    df = pd.DataFrame({'a': [1, 2, 3, 4]})
    scaler = DynamicScaler(min_post_std=100, scoring=lambda _, arr: arr.std())
    scaler.fit(df)
    assert scaler.report_['a']['chosen_scaler'] == 'None'
    assert scaler.report_['a']['reason'] == 'all_rejected'


def test_scoring_improves():
    np.random.seed(0)
    df = pd.DataFrame({'a': np.exp(np.random.normal(size=100))})
    scaler = DynamicScaler(random_state=0)
    scaler.fit(df)
    assert scaler.report_['a']['chosen_scaler'] == 'PowerTransformer'
