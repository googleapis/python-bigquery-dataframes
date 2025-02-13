# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import db_dtypes  # type: ignore
import geopandas as gpd  # type: ignore
import pandas as pd
import pyarrow as pa
import pytest

import bigframes.bigquery as bbq
import bigframes.dtypes
import bigframes.pandas as bpd


@pytest.mark.parametrize(
    ("json_path", "expected_json"),
    [
        pytest.param("$.a", [{"a": 10}], id="simple"),
        pytest.param("$.a.b.c", [{"a": {"b": {"c": 10, "d": []}}}], id="nested"),
    ],
)
def test_json_set_at_json_path(json_path, expected_json):
    original_json = [{"a": {"b": {"c": "tester", "d": []}}}]
    s = bpd.Series(original_json, dtype=db_dtypes.JSONDtype())
    actual = bbq.json_set(s, json_path_value_pairs=[(json_path, 10)])

    expected = bpd.Series(expected_json, dtype=db_dtypes.JSONDtype())
    pd.testing.assert_series_equal(
        actual.to_pandas(),
        expected.to_pandas(),
    )


@pytest.mark.parametrize(
    ("json_value", "expected_json"),
    [
        pytest.param(10, [{"a": {"b": 10}}, {"a": {"b": 10}}], id="int"),
        pytest.param(0.333, [{"a": {"b": 0.333}}, {"a": {"b": 0.333}}], id="float"),
        pytest.param("eng", [{"a": {"b": "eng"}}, {"a": {"b": "eng"}}], id="string"),
        pytest.param([1, 2], [{"a": {"b": 1}}, {"a": {"b": 2}}], id="series"),
    ],
)
def test_json_set_at_json_value_type(json_value, expected_json):
    original_json = [{"a": {"b": "dev"}}, {"a": {"b": [1, 2]}}]
    s = bpd.Series(original_json, dtype=db_dtypes.JSONDtype())
    actual = bbq.json_set(s, json_path_value_pairs=[("$.a.b", json_value)])

    expected = bpd.Series(expected_json, dtype=db_dtypes.JSONDtype())
    pd.testing.assert_series_equal(
        actual.to_pandas(),
        expected.to_pandas(),
    )


def test_json_set_w_more_pairs():
    original_json = [{"a": 2}, {"b": 5}, {"c": 1}]
    s = bpd.Series(original_json, dtype=db_dtypes.JSONDtype())
    actual = bbq.json_set(
        s, json_path_value_pairs=[("$.a", 1), ("$.b", 2), ("$.a", [3, 4, 5])]
    )

    expected_json = [{"a": 3, "b": 2}, {"a": 4, "b": 2}, {"a": 5, "b": 2, "c": 1}]
    expected = bpd.Series(expected_json, dtype=db_dtypes.JSONDtype())
    pd.testing.assert_series_equal(
        actual.to_pandas(),
        expected.to_pandas(),
    )


def test_json_set_w_invalid_json_path_value_pairs():
    s = bpd.Series([{"a": 10}], dtype=db_dtypes.JSONDtype())
    with pytest.raises(ValueError):
        bbq.json_set(s, json_path_value_pairs=[("$.a", 1, 100)])  # type: ignore


def test_json_set_w_invalid_value_type():
    s = bpd.Series([{"a": 10}], dtype=db_dtypes.JSONDtype())
    with pytest.raises(TypeError):
        bbq.json_set(
            s,
            json_path_value_pairs=[
                (
                    "$.a",
                    bpd.read_pandas(
                        gpd.GeoSeries.from_wkt(["POINT (1 2)", "POINT (2 1)"])
                    ),
                )
            ],
        )


def test_json_set_w_invalid_series_type():
    with pytest.raises(TypeError):
        bbq.json_set(bpd.Series([1, 2]), json_path_value_pairs=[("$.a", 1)])


def test_json_extract_from_json():
    s = bpd.Series(
        [{"a": {"b": [1, 2]}}, {"a": {"c": 1}}, {"a": {"b": 0}}],
        dtype=db_dtypes.JSONDtype(),
    )
    actual = bbq.json_extract(s, "$.a.b").to_pandas()
    expected = bpd.Series([[1, 2], None, 0], dtype=db_dtypes.JSONDtype()).to_pandas()
    pd.testing.assert_series_equal(
        actual,
        expected,
    )


def test_json_extract_from_string():
    s = bpd.Series(
        ['{"a": {"b": [1, 2]}}', '{"a": {"c": 1}}', '{"a": {"b": 0}}'],
        dtype=pd.StringDtype(storage="pyarrow"),
    )
    actual = bbq.json_extract(s, "$.a.b")
    expected = bpd.Series(["[1,2]", None, "0"], dtype=pd.StringDtype(storage="pyarrow"))
    pd.testing.assert_series_equal(
        actual.to_pandas(),
        expected.to_pandas(),
    )


def test_json_extract_w_invalid_series_type():
    with pytest.raises(TypeError):
        bbq.json_extract(bpd.Series([1, 2]), "$.a")


def test_json_extract_array_from_json():
    s = bpd.Series(
        [{"a": ["ab", "2", "3 xy"]}, {"a": []}, {"a": ["4", "5"]}, {}],
        dtype=db_dtypes.JSONDtype(),
    )
    actual = bbq.json_extract_array(s, "$.a")

    # This code provides a workaround for issue https://github.com/apache/arrow/issues/45262,
    # which currently prevents constructing a series using the pa.list_(db_types.JSONArrrowType())
    sql = """
        SELECT 0 AS id, [JSON '"ab"', JSON '"2"', JSON '"3 xy"'] AS data,
        UNION ALL
        SELECT 1, [],
        UNION ALL
        SELECT 2, [JSON '"4"', JSON '"5"'],
        UNION ALL
        SELECT 3, null,
    """
    df = bpd.read_gbq(sql).set_index("id").sort_index()
    expected = df["data"]
    expected.index.name = None
    expected.name = None

    pd.testing.assert_series_equal(
        actual.to_pandas(),
        expected.to_pandas(),
    )


def test_json_extract_array_from_json_strings():
    s = bpd.Series(
        ['{"a": ["ab", "2", "3 xy"]}', '{"a": []}', '{"a": ["4","5"]}', "{}"],
        dtype=pd.StringDtype(storage="pyarrow"),
    )
    actual = bbq.json_extract_array(s, "$.a")
    expected = bpd.Series(
        [['"ab"', '"2"', '"3 xy"'], [], ['"4"', '"5"'], None],
        dtype=pd.ArrowDtype(pa.list_(pa.string())),
    )
    pd.testing.assert_series_equal(
        actual.to_pandas(),
        expected.to_pandas(),
    )


def test_json_extract_array_from_json_array_strings():
    s = bpd.Series(
        ["[1, 2, 3]", "[]", "[4,5]"],
        dtype=pd.StringDtype(storage="pyarrow"),
    )
    actual = bbq.json_extract_array(s)
    expected = bpd.Series(
        [["1", "2", "3"], [], ["4", "5"]],
        dtype=pd.ArrowDtype(pa.list_(pa.string())),
    )
    pd.testing.assert_series_equal(
        actual.to_pandas(),
        expected.to_pandas(),
    )


def test_json_extract_array_w_invalid_series_type():
    s = bpd.Series([1, 2])
    with pytest.raises(TypeError):
        bbq.json_extract_array(s)


def test_json_extract_string_array_from_json_strings():
    s = bpd.Series(['{"a": ["ab", "2", "3 xy"]}', '{"a": []}', '{"a": ["4","5"]}'])
    actual = bbq.json_extract_string_array(s, "$.a")
    expected = bpd.Series([["ab", "2", "3 xy"], [], ["4", "5"]])
    pd.testing.assert_series_equal(
        actual.to_pandas(),
        expected.to_pandas(),
    )


def test_json_extract_string_array_from_array_strings():
    s = bpd.Series(["[1, 2, 3]", "[]", "[4,5]"])
    actual = bbq.json_extract_string_array(s)
    expected = bpd.Series([["1", "2", "3"], [], ["4", "5"]])
    pd.testing.assert_series_equal(
        actual.to_pandas(),
        expected.to_pandas(),
    )


def test_json_extract_string_array_as_float_array_from_array_strings():
    s = bpd.Series(["[1, 2.5, 3]", "[]", "[4,5]"])
    actual = bbq.json_extract_string_array(s, value_dtype=bigframes.dtypes.FLOAT_DTYPE)
    expected = bpd.Series([[1, 2.5, 3], [], [4, 5]])
    pd.testing.assert_series_equal(
        actual.to_pandas(),
        expected.to_pandas(),
    )


def test_json_extract_string_array_w_invalid_series_type():
    with pytest.raises(TypeError):
        bbq.json_extract_string_array(bpd.Series([1, 2]))


def test_parse_json_w_invalid_series_type():
    with pytest.raises(TypeError):
        bbq.parse_json(bpd.Series([1, 2]))
