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

import pandas as pd
import pytest

import bigframes as bf
import bigframes.display.html as bf_html


@pytest.mark.parametrize(
    ("data", "expected_alignments", "expected_strings"),
    [
        (
            {
                "string_col": ["a", "b", "c"],
                "int_col": [1, 2, 3],
                "float_col": [1.1, 2.2, 3.3],
                "bool_col": [True, False, True],
            },
            {
                "string_col": "left",
                "int_col": "right",
                "float_col": "right",
                "bool_col": "left",
            },
            ["1.100000", "2.200000", "3.300000"],
        ),
    ],
)
def test_render_html_alignment_and_precision(
    data, expected_alignments, expected_strings
):
    df = pd.DataFrame(data)
    html = bf_html.render_html(dataframe=df, table_id="test-table")

    for _, align in expected_alignments.items():
        assert 'th style="text-align: left;"' in html
        assert f'<td style="text-align: {align};">' in html

    for expected_string in expected_strings:
        assert expected_string in html


def test_render_html_precision():
    data = {"float_col": [3.14159265]}
    df = pd.DataFrame(data)

    with bf.option_context("display.precision", 4):
        html = bf_html.render_html(dataframe=df, table_id="test-table")
        assert "3.1416" in html

    # Make sure we reset to default
    html = bf_html.render_html(dataframe=df, table_id="test-table")
    assert "3.141593" in html
