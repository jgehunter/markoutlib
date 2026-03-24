"""Tests for pandas <-> polars conversion."""

import polars as pl
import pytest


def test_polars_passthrough():
    from markoutlib._compat import maybe_to_polars

    df = pl.DataFrame({"a": [1, 2, 3]})
    result = maybe_to_polars(df)
    assert result is df


def test_pandas_conversion():
    pd = pytest.importorskip("pandas")
    from markoutlib._compat import maybe_to_polars

    pdf = pd.DataFrame({"a": [1, 2, 3]})
    result = maybe_to_polars(pdf)
    assert isinstance(result, pl.DataFrame)
    assert result["a"].to_list() == [1, 2, 3]


def test_invalid_type_raises():
    from markoutlib._compat import maybe_to_polars

    with pytest.raises(TypeError, match="expected polars or pandas DataFrame"):
        maybe_to_polars({"a": [1]})


def test_to_pandas_without_pandas(monkeypatch):
    from markoutlib import _compat

    monkeypatch.setattr(_compat, "_get_pandas", lambda: None)
    with pytest.raises(ImportError, match="pip install markoutlib"):
        _compat.to_pandas(pl.DataFrame({"a": [1]}))


def test_to_pandas_roundtrip():
    pd = pytest.importorskip("pandas")
    from markoutlib._compat import maybe_to_polars, to_pandas

    pdf = pd.DataFrame({"a": [1.0, 2.0], "b": ["x", "y"]})
    pldf = maybe_to_polars(pdf)
    result = to_pandas(pldf)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["a", "b"]
