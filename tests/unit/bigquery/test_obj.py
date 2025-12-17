# Copyright 2025 Google LLC
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
from unittest.mock import MagicMock

import bigframes.bigquery.obj as obj
import bigframes.operations as ops
import bigframes.series as series

def test_fetch_metadata_op_structure():
    op = ops.obj_fetch_metadata_op
    assert op.name == "obj_fetch_metadata"

def test_get_access_url_op_structure():
    op = ops.ObjGetAccessUrl(mode="r")
    assert op.name == "obj_get_access_url"
    assert op.mode == "r"

def test_get_access_url_with_duration_op_structure():
    op = ops.ObjGetAccessUrlWithDuration(mode="rw")
    assert op.name == "obj_get_access_url_with_duration"
    assert op.mode == "rw"

def test_make_ref_op_structure():
    op = ops.obj_make_ref_op
    assert op.name == "obj_make_ref"

def test_make_ref_json_op_structure():
    op = ops.obj_make_ref_json_op
    assert op.name == "obj_make_ref_json"

def test_fetch_metadata_calls_apply_unary_op():
    s = MagicMock(spec=series.Series)

    obj.fetch_metadata(s)

    s._apply_unary_op.assert_called_once()
    args, _ = s._apply_unary_op.call_args
    assert args[0] == ops.obj_fetch_metadata_op

def test_get_access_url_calls_apply_unary_op_without_duration():
    s = MagicMock(spec=series.Series)

    obj.get_access_url(s, mode="r")

    s._apply_unary_op.assert_called_once()
    args, _ = s._apply_unary_op.call_args
    assert isinstance(args[0], ops.ObjGetAccessUrl)
    assert args[0].mode == "r"

def test_get_access_url_calls_apply_binary_op_with_duration():
    s = MagicMock(spec=series.Series)
    duration = MagicMock(spec=series.Series)

    obj.get_access_url(s, mode="rw", duration=duration)

    s._apply_binary_op.assert_called_once()
    args, kwargs = s._apply_binary_op.call_args
    assert args[0] == duration
    assert isinstance(args[1], ops.ObjGetAccessUrlWithDuration)
    assert args[1].mode == "rw"

def test_make_ref_calls_apply_binary_op_with_authorizer():
    uri = MagicMock(spec=series.Series)
    auth = MagicMock(spec=series.Series)

    obj.make_ref(uri, authorizer=auth)

    uri._apply_binary_op.assert_called_once()
    args, _ = uri._apply_binary_op.call_args
    assert args[0] == auth
    assert args[1] == ops.obj_make_ref_op

def test_make_ref_calls_apply_unary_op_without_authorizer():
    json_val = MagicMock(spec=series.Series)

    obj.make_ref(json_val)

    json_val._apply_unary_op.assert_called_once()
    args, _ = json_val._apply_unary_op.call_args
    assert args[0] == ops.obj_make_ref_json_op
