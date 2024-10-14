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

import pyarrow as pa
import pytest

import bigframes.pandas as bpd


@pytest.mark.parametrize(
    ("key", "should_warn"),
    [
        pytest.param(0, False, id="non_string_key_should_not_warn"),
        pytest.param("a", True, id="string_key_should_warn"),
    ],
)
def test_non_string_indexed_series_struct_accessor_warning(session, key, should_warn):
    s = bpd.Series(
        [
            {"project": "pandas", "version": 1},
        ],
        dtype=bpd.ArrowDtype(
            pa.struct([("project", pa.string()), ("version", pa.int64())])
        ),
        session=session,
    )

    if should_warn:
        with pytest.warns(UserWarning, match=r"Series\.struct\.field\(.+\)"):
            s[key]
    else:
        s[key]


@pytest.mark.parametrize(
    "key",
    [
        pytest.param(0, id="non_string_key"),
        pytest.param("a", id="string_key"),
    ],
)
def test_string_indexed_series_struct_accessor_no_warning(session, key):
    s = bpd.Series(
        [
            {"project": "pandas", "version": 1},
        ],
        dtype=bpd.ArrowDtype(
            pa.struct([("project", pa.string()), ("version", pa.int64())])
        ),
        index=["p1"],
        session=session,
    )

    s[key]
