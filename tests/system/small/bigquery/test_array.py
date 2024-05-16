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

import numpy as np
import pandas as pd
import pytest

import bigframes.bigquery as bbq
import bigframes.pandas as bpd


def test_array_length():
    series = bpd.Series([["A", "AA", "AAA"], ["BB", "B"], np.nan, [], ["C"]])
    # TODO(b/336880368): Allow for NULL values to be input for ARRAY columns.
    # Once we actually store NULL values, this will be NULL where the input is NULL.
    expected = bpd.Series([3, 2, 0, 0, 1])
    pd.testing.assert_series_equal(
        bbq.array_length(series).to_pandas(),
        expected.to_pandas(),
    )


@pytest.mark.parametrize(
    ("input_data", "output_data"),
    [
        pytest.param([1, 2, 3, 4, 5], [[1, 2], [3, 4], [5]], id="ints"),
        pytest.param(
            ["e", "d", "c", "b", "a"],
            [["e", "d"], ["c", "b"], ["a"]],
            id="reverse_strings",
        ),
        pytest.param(
            [1.0, 2.0, np.nan, np.nan, np.nan], [[1.0, 2.0], [], []], id="nans"
        ),
        pytest.param(
            [{"A": {"x": 1.0}}, {"A": {"z": 4.0}}, {}, {"B": "b"}, np.nan],
            [[{"A": {"x": 1.0}}, {"A": {"z": 4.0}}], [{}, {"B": "b"}], []],
            id="structs",
        ),
    ],
)
def test_array_agg_w_series(input_data, output_data):
    input_index = ["a", "a", "b", "b", "c"]
    series = bpd.Series(input_data, index=input_index)
    result = bbq.array_agg(series.groupby(level=0))

    expected = bpd.Series(output_data, index=["a", "b", "c"])
    pd.testing.assert_series_equal(
        result.to_pandas(),
        expected.to_pandas(),
    )


def test_array_agg_w_dataframe():
    data = {
        "a": [1, 1, 2, 1],
        "b": [2, None, 1, 2],
        "c": [3, 4, 3, 2],
    }
    df = bpd.DataFrame(data)
    result = bbq.array_agg(df.groupby(by=["b"]))

    expected_data = {
        "b": [1.0, 2.0],
        "a": [[2], [1, 1]],
        "c": [[3], [3, 2]],
    }
    expected = bpd.DataFrame(expected_data).set_index("b")

    pd.testing.assert_frame_equal(
        result.to_pandas(),
        expected.to_pandas(),
    )

def assert_array_agg_matches_after_explode():
    data = {
        "index": np.arange(10),
        "a": [np.random.randint(0, 10, 10) for _ in range(10)],
        "b": [np.random.randint(0, 10, 10) for _ in range(10)],
    }
    df = bpd.DataFrame(data).set_index("index")
    result = bbq.array_agg(df.explode(["a", "b"]).groupby(level=0))
    result.index.name = "index"

    pd.testing.assert_frame_equal(
        result.to_pandas(),
        df.to_pandas(),
    )
