from unittest.mock import Mock

import pytest

from . import resources


@pytest.fixture
def mock_df(monkeypatch: pytest.MonkeyPatch):
    dataframe = resources.create_dataframe(monkeypatch)
    monkeypatch.setattr(dataframe, "to_pandas", Mock())
    return dataframe


@pytest.mark.parametrize(
    "api_name, kwargs",
    [
        ("to_csv", {"allow_large_results": True}),
        ("to_json", {"allow_large_results": True}),
        ("to_numpy", {"allow_large_results": True}),
        ("to_parquet", {"allow_large_results": True}),
        ("to_dict", {"allow_large_results": True}),
        ("to_excel", {"excel_writer": "abc", "allow_large_results": True}),
        ("to_latex", {"allow_large_results": True}),
        ("to_records", {"allow_large_results": True}),
        ("to_string", {"allow_large_results": True}),
        ("to_html", {"allow_large_results": True}),
        ("to_markdown", {"allow_large_results": True}),
        ("to_pickle", {"path": "abc", "allow_large_results": True}),
        ("to_orc", {"allow_large_results": True}),
    ],
)
def test_dataframe_to_pandas(mock_df, api_name, kwargs):
    getattr(mock_df, api_name)(**kwargs)
    mock_df.to_pandas.assert_called_once_with(
        allow_large_results=kwargs["allow_large_results"]
    )
