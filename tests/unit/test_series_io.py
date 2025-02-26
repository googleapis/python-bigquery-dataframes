from unittest.mock import Mock

import pytest

from . import resources


@pytest.fixture
def mock_series(monkeypatch: pytest.MonkeyPatch):
    dataframe = resources.create_dataframe(monkeypatch)
    series = dataframe["col"]
    monkeypatch.setattr(series, "to_pandas", Mock())
    return series


@pytest.mark.parametrize(
    "api_name, kwargs",
    [
        ("to_csv", {"allow_large_results": True}),
        ("to_dict", {"allow_large_results": True}),
        ("to_excel", {"excel_writer": "abc", "allow_large_results": True}),
        ("to_json", {"allow_large_results": True}),
        ("to_latex", {"allow_large_results": True}),
        ("to_list", {"allow_large_results": True}),
        ("to_markdown", {"allow_large_results": True}),
        ("to_numpy", {"allow_large_results": True}),
        ("to_pickle", {"path": "abc", "allow_large_results": True}),
        ("to_string", {"allow_large_results": True}),
        ("to_xarray", {"allow_large_results": True}),
    ],
)
def test_series_allow_large_results_param_passing(mock_series, api_name, kwargs):
    getattr(mock_series, api_name)(**kwargs)
    mock_series.to_pandas.assert_called_once_with(
        allow_large_results=kwargs["allow_large_results"]
    )
