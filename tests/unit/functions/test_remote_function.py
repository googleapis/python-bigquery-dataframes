# Copyright 2023 Google LLC
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

import re

import pandas
import pytest

import bigframes.functions.function as bff
import bigframes.series
from bigframes.testing import mocks


@pytest.mark.parametrize(
    "series_type",
    (
        pytest.param(
            pandas.Series,
            id="pandas.Series",
        ),
        pytest.param(
            bigframes.series.Series,
            id="bigframes.series.Series",
        ),
    ),
)
def test_series_input_types_to_str(series_type):
    """Check that is_row_processor=True uses str as the input type to serialize a row."""
    session = mocks.create_bigquery_session()
    remote_function_decorator = bff.remote_function(
        session=session, cloud_function_service_account="default"
    )

    with pytest.warns(
        bigframes.exceptions.PreviewWarning,
        match=re.escape("input_types=Series is in preview."),
    ):

        @remote_function_decorator
        def axis_1_function(myparam: series_type) -> str:  # type: ignore
            return "Hello, " + myparam["str_col"] + "!"  # type: ignore

    # Still works as a normal function.
    assert axis_1_function(pandas.Series({"str_col": "World"})) == "Hello, World!"


def test_missing_input_types():
    session = mocks.create_bigquery_session()
    remote_function_decorator = bff.remote_function(
        session=session, cloud_function_service_account="default"
    )

    def function_without_parameter_annotations(myparam) -> str:
        return str(myparam)

    assert function_without_parameter_annotations(42) == "42"

    with pytest.raises(
        ValueError,
        match="'input_types' was not set .* 'myparam' is missing a type annotation",
    ):
        remote_function_decorator(function_without_parameter_annotations)


def test_missing_output_type():
    session = mocks.create_bigquery_session()
    remote_function_decorator = bff.remote_function(
        session=session, cloud_function_service_account="default"
    )

    def function_without_return_annotation(myparam: int):
        return str(myparam)

    assert function_without_return_annotation(42) == "42"

    with pytest.raises(
        ValueError,
        match="'output_type' was not set .* missing a return type annotation",
    ):
        remote_function_decorator(function_without_return_annotation)


# --- Tests for bpd.deploy_remote_function ---
@mock.patch("bigframes.functions._function_session.FunctionSession.remote_function")
def test_bpd_deploy_remote_function_calls_session_remote_function_deploy_true(mock_session_remote_function):
    mock_session = mocks.create_bigquery_session()
    # The decorator @bpd.deploy_remote_function itself will call the mocked session method.
    # The mock_session_remote_function is what bpd.deploy_remote_function will eventually call on the session.

    @bpd.deploy_remote_function(session=mock_session, cloud_function_service_account="test_sa@example.com")
    def my_remote_func(x: int) -> int:
        return x * 2

    mock_session_remote_function.assert_called_once()
    # Check some key args passed to the session's remote_function method
    args, kwargs = mock_session_remote_function.call_args
    assert kwargs.get("deploy_immediately") is True
    assert kwargs.get("reuse") is True  # Default reuse is True
    assert kwargs.get("name") is None # Default name is None

    # Test that the function is still callable locally (it calls the original python func, not the mock)
    assert my_remote_func(10) == 20


@mock.patch("bigframes.functions._function_session.FunctionSession.remote_function")
def test_bpd_deploy_remote_function_no_reuse_calls_session_remote_function_deploy_true(mock_session_remote_function):
    mock_session = mocks.create_bigquery_session()

    @bpd.deploy_remote_function(session=mock_session, cloud_function_service_account="test_sa@example.com", reuse=False)
    def my_remote_func_no_reuse(x: int) -> int:
        return x * 3

    mock_session_remote_function.assert_called_once()
    args, kwargs = mock_session_remote_function.call_args
    assert kwargs.get("deploy_immediately") is True
    assert kwargs.get("reuse") is False

    assert my_remote_func_no_reuse(5) == 15


@mock.patch("bigframes.functions._function_session.FunctionSession.remote_function")
def test_bpd_deploy_remote_function_with_name_calls_session_remote_function_deploy_true(mock_session_remote_function):
    mock_session = mocks.create_bigquery_session()

    @bpd.deploy_remote_function(
        session=mock_session,
        cloud_function_service_account="test_sa@example.com",
        name="custom_name"
    )
    def my_named_remote_func(x: int) -> int:
        return x * 4

    mock_session_remote_function.assert_called_once()
    args, kwargs = mock_session_remote_function.call_args
    assert kwargs.get("deploy_immediately") is True
    assert kwargs.get("name") == "custom_name"

    assert my_named_remote_func(3) == 12


# --- Tests for bpd.remote_function (checking deploy_immediately=False) ---
@mock.patch("bigframes.functions._function_session.FunctionSession.remote_function")
def test_bpd_remote_function_calls_session_remote_function_deploy_false(mock_session_remote_function):
    mock_session = mocks.create_bigquery_session()

    @bpd.remote_function(session=mock_session, cloud_function_service_account="test_sa@example.com")
    def my_std_remote_func(x: int) -> int:
        return x * 5

    mock_session_remote_function.assert_called_once()
    args, kwargs = mock_session_remote_function.call_args
    # Check that deploy_immediately is False or not passed (relying on default)
    assert kwargs.get("deploy_immediately", False) is False # Checks if key exists and is False, or defaults to False if key not present

    assert my_std_remote_func(6) == 30


# --- Tests for bpd.deploy_udf ---
@mock.patch("bigframes.functions._function_session.FunctionSession.udf")
def test_bpd_deploy_udf_calls_session_udf_deploy_true(mock_session_udf_method):
    mock_session = mocks.create_bigquery_session(
        default_project="test-project",
        default_location="us-central1"
    )

    @bpd.deploy_udf(session=mock_session, dataset="my_dataset", name="my_udf_1")
    def my_udf(y: str) -> str:
        return f"hello {y}"

    mock_session_udf_method.assert_called_once()
    args, kwargs = mock_session_udf_method.call_args
    assert kwargs.get("deploy_immediately") is True
    assert kwargs.get("name") == "my_udf_1"
    assert kwargs.get("dataset") == "my_dataset"

    assert my_udf("world") == "hello world"


@mock.patch("bigframes.functions._function_session.FunctionSession.udf")
def test_bpd_deploy_udf_no_name_calls_session_udf_deploy_true(mock_session_udf_method):
    mock_session = mocks.create_bigquery_session(
        default_project="test-project",
        default_location="us-central1"
    )

    @bpd.deploy_udf(session=mock_session, dataset="my_dataset") # No explicit name
    def my_anon_udf(val: float) -> float:
        return val + 1.0

    mock_session_udf_method.assert_called_once()
    args, kwargs = mock_session_udf_method.call_args
    assert kwargs.get("deploy_immediately") is True
    assert kwargs.get("name") is None # Name should be None, letting system generate

    assert my_anon_udf(1.5) == 2.5


@mock.patch("bigframes.functions._function_session.FunctionSession.udf")
def test_bpd_deploy_udf_with_default_dataset_calls_session_udf_deploy_true(mock_session_udf_method):
    mock_session = mocks.create_bigquery_session(
        default_project="test-project",
        default_location="us-central1"
    )
    # We expect the dataset to be resolved to session's default by the time
    # FunctionSession.udf is called if not provided explicitly.

    @bpd.deploy_udf(session=mock_session, name="my_udf_2") # dataset should come from session
    def my_udf_default_ds(z: bytes) -> bytes:
        return z + b" extra"

    mock_session_udf_method.assert_called_once()
    args, kwargs = mock_session_udf_method.call_args
    assert kwargs.get("deploy_immediately") is True
    assert kwargs.get("name") == "my_udf_2"
    # The `dataset` kwarg might be None here if it's resolved from session later,
    # or it might be pre-resolved. The key is that the session method is called.
    # The actual dataset resolution logic is part of FunctionSession.udf, not this bpd wrapper.
    # So, we check that `dataset` is passed as None or the expected default from session context.
    # For this test, we'll assume the bpd wrapper passes `dataset=None` if not specified.
    assert kwargs.get("dataset") is None


    assert my_udf_default_ds(b"data") == b"data extra"


# --- Tests for bpd.udf (checking deploy_immediately=False) ---
@mock.patch("bigframes.functions._function_session.FunctionSession.udf")
def test_bpd_udf_calls_session_udf_deploy_false(mock_session_udf_method):
    mock_session = mocks.create_bigquery_session(
        default_project="test-project",
        default_location="us-central1"
    )

    @bpd.udf(session=mock_session, dataset="my_dataset_std", name="my_std_udf")
    def my_std_udf(y: str) -> str:
        return f"standard hello {y}"

    mock_session_udf_method.assert_called_once()
    args, kwargs = mock_session_udf_method.call_args
    assert kwargs.get("deploy_immediately", False) is False
    assert kwargs.get("name") == "my_std_udf"
    assert kwargs.get("dataset") == "my_dataset_std"

    assert my_std_udf("dev") == "standard hello dev"
