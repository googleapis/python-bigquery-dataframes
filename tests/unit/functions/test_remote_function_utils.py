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

import pytest

import bigframes.dtypes
from bigframes.functions import _utils


@pytest.mark.parametrize(
    ["metadata_options", "metadata_string"],
    (
        pytest.param(
            {},
            '{"value": {}}',
            id="empty",
        ),
        pytest.param(
            {"python_output_type": None},
            '{"value": {}}',
            id="None",
        ),
        pytest.param(
            {"python_output_type": bool},
            '{"value": {"python_output_type": "bool"}}',
            id="bool",
        ),
        pytest.param(
            {"python_output_type": bytes},
            '{"value": {"python_output_type": "bytes"}}',
            id="bytes",
        ),
        pytest.param(
            {"python_output_type": float},
            '{"value": {"python_output_type": "float"}}',
            id="float",
        ),
        pytest.param(
            {"python_output_type": int},
            '{"value": {"python_output_type": "int"}}',
            id="int",
        ),
        pytest.param(
            {"python_output_type": str},
            '{"value": {"python_output_type": "str"}}',
            id="str",
        ),
        pytest.param(
            {"python_output_type": list[bool]},
            '{"value": {"python_output_type": "list[bool]"}}',
            id="list-bool",
        ),
        pytest.param(
            {"python_output_type": list[float]},
            '{"value": {"python_output_type": "list[float]"}}',
            id="list-float",
        ),
        pytest.param(
            {"python_output_type": list[int]},
            '{"value": {"python_output_type": "list[int]"}}',
            id="list-int",
        ),
        pytest.param(
            {"python_output_type": list[str]},
            '{"value": {"python_output_type": "list[str]"}}',
            id="list-str",
        ),
    ),
)
def test_get_bigframes_metadata(metadata_options, metadata_string):
    assert _utils.get_bigframes_metadata(**metadata_options) == metadata_string


@pytest.mark.parametrize(
    ["metadata_string", "python_output_type"],
    (
        pytest.param(
            None,
            None,
            id="None",
        ),
        pytest.param(
            "",
            None,
            id="empty",
        ),
        pytest.param(
            "{}",
            None,
            id="empty-dict",
        ),
        pytest.param(
            '{"value": {}}',
            None,
            id="empty-value",
        ),
        pytest.param(
            '{"value": {"python_output_type": "blah"}}',
            None,
            id="invalid-type",
        ),
        pytest.param(
            '{"value": {"python_output_type": "bool"}}',
            bool,
            id="bool",
        ),
        pytest.param(
            '{"value": {"python_output_type": "float"}}',
            float,
            id="float",
        ),
        pytest.param(
            '{"value": {"python_output_type": "int"}}',
            int,
            id="int",
        ),
        pytest.param(
            '{"value": {"python_output_type": "str"}}',
            str,
            id="str",
        ),
        pytest.param(
            '{"value": {"python_output_type": "list[bool]"}}',
            list[bool],
            id="list-bool",
        ),
        pytest.param(
            '{"value": {"python_output_type": "list[float]"}}',
            list[float],
            id="list-float",
        ),
        pytest.param(
            '{"value": {"python_output_type": "list[int]"}}',
            list[int],
            id="list-int",
        ),
        pytest.param(
            '{"value": {"python_output_type": "list[str]"}}',
            list[str],
            id="list-str",
        ),
    ),
)
def test_get_python_output_type_from_bigframes_metadata(
    metadata_string, python_output_type
):
    assert (
        _utils.get_python_output_type_from_bigframes_metadata(metadata_string)
        == python_output_type
    )


def test_metadata_roundtrip_supported_array_types():
    for array_of in bigframes.dtypes.RF_SUPPORTED_ARRAY_OUTPUT_PYTHON_TYPES:
        ser = _utils.get_bigframes_metadata(python_output_type=list[array_of])  # type: ignore
        deser = _utils.get_python_output_type_from_bigframes_metadata(ser)
        assert deser == list[array_of]  # type: ignore
