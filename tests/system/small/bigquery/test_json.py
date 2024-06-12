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

import json

import pandas as pd

import bigframes.bigquery as bbq
import bigframes.pandas as bpd


def _get_series_from_json(json_data):
    sql = " UNION ALL ".join(
        [f"SELECT JSON '{json.dumps(data)}' AS data" for data in json_data]
    )
    return bpd.read_gbq(sql)["data"]


def test_json_set():
    init_json = [
        {"a": 1},
    ]
    s = _get_series_from_json(init_json)
    actual = bbq.json_set(s, json_path_value_pairs=[("$.a", 10)])

    expected_json = [
        {"a": 10},
    ]
    expected = _get_series_from_json(expected_json)
    pd.testing.assert_series_equal(
        actual.to_pandas(),
        expected.to_pandas(),
    )


def test_json_set_w_nested_json():
    init_json = [
        {"a": {"b": {"c": "tester", "d": []}}},
    ]
    s = _get_series_from_json(init_json)
    actual = bbq.json_set(s, json_path_value_pairs=[("$.a.b.c", "user")])

    expected_json = [
        {"a": {"b": {"c": "user", "d": []}}},
    ]
    expected = _get_series_from_json(expected_json)
    pd.testing.assert_series_equal(
        actual.to_pandas(),
        expected.to_pandas(),
    )


def test_json_set_w_ordered_pairs():
    init_json = [
        {"a": {"b": {"c": {}}}},
    ]
    s = _get_series_from_json(init_json)
    actual = bbq.json_set(
        s, json_path_value_pairs=[("$.a.b.e", "user"), ("$.a.b.e", "dev")]
    )

    expected_json = [
        {"a": {"b": {"c": {}, "e": "dev"}}},
    ]
    expected = _get_series_from_json(expected_json)
    pd.testing.assert_series_equal(
        actual.to_pandas(),
        expected.to_pandas(),
    )
