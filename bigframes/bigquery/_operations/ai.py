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

"""This module integrates BigQuery built-in AI functions for use with Series/DataFrame objects,
such as AI.GENERATE_BOOL:
https://cloud.google.com/bigquery/docs/reference/standard-sql/bigqueryml-syntax-ai-generate-bool"""

from __future__ import annotations

import json
from typing import Any, List, Literal, Mapping, Tuple, Union

import pandas as pd

from bigframes import clients, dtypes, series, session
from bigframes.core import convert, log_adapter
from bigframes.operations import ai_ops

PROMPT_TYPE = Union[
    series.Series,
    pd.Series,
    List[Union[str, series.Series, pd.Series]],
    Tuple[Union[str, series.Series, pd.Series], ...],
]


@log_adapter.method_logger(custom_base_name="bigquery_ai")
def generate_bool(
    prompt: PROMPT_TYPE,
    *,
    connection_id: str | None = None,
    endpoint: str | None = None,
    request_type: Literal["dedicated", "shared", "unspecified"] = "unspecified",
    model_params: Mapping[Any, Any] | None = None,
) -> series.Series:
    """
    Returns the AI analysis based on the prompt, which can be any combination of text and unstructured data.

    **Examples:**

        >>> import bigframes.pandas as bpd
        >>> import bigframes.bigquery as bbq
        >>> bpd.options.display.progress_bar = None
        >>> df = bpd.DataFrame({
        ...     "col_1": ["apple", "bear", "pear"],
        ...     "col_2": ["fruit", "animal", "animal"]
        ... })
        >>> bbq.ai.generate_bool((df["col_1"], " is a ", df["col_2"]))
        0    {'result': True, 'full_response': '{"candidate...
        1    {'result': True, 'full_response': '{"candidate...
        2    {'result': False, 'full_response': '{"candidat...
        dtype: struct<result: bool, full_response: extension<dbjson<JSONArrowType>>, status: string>[pyarrow]

        >>> bbq.ai.generate_bool((df["col_1"], " is a ", df["col_2"])).struct.field("result")
        0     True
        1     True
        2    False
        Name: result, dtype: boolean

    Args:
        prompt (Series | List[str|Series] | Tuple[str|Series, ...]):
            A mixture of Series and string literals that specifies the prompt to send to the model. The Series can be BigFrames Series
            or pandas Series.
        connection_id (str, optional):
            Specifies the connection to use to communicate with the model. For example, `myproject.us.myconnection`.
            If not provided, the connection from the current session will be used.
        endpoint (str, optional):
            Specifies the Vertex AI endpoint to use for the model. For example `"gemini-2.5-flash"`. You can specify any
            generally available or preview Gemini model. If you specify the model name, BigQuery ML automatically identifies and
            uses the full endpoint of the model. If you don't specify an ENDPOINT value, BigQuery ML selects a recent stable
            version of Gemini to use.
        request_type (Literal["dedicated", "shared", "unspecified"]):
            Specifies the type of inference request to send to the Gemini model. The request type determines what quota the request uses.
            * "dedicated": function only uses Provisioned Throughput quota. The function returns the error Provisioned throughput is not
            purchased or is not active if Provisioned Throughput quota isn't available.
            * "shared": the function only uses dynamic shared quota (DSQ), even if you have purchased Provisioned Throughput quota.
            * "unspecified": If you haven't purchased Provisioned Throughput quota, the function uses DSQ quota.
            If you have purchased Provisioned Throughput quota, the function uses the Provisioned Throughput quota first.
            If requests exceed the Provisioned Throughput quota, the overflow traffic uses DSQ quota.
        model_params (Mapping[Any, Any]):
            Provides additional parameters to the model. The MODEL_PARAMS value must conform to the generateContent request body format.

    Returns:
        bigframes.series.Series: A new struct Series with the result data. The struct contains these fields:
        * "result": a BOOL value containing the model's response to the prompt. The result is None if the request fails or is filtered by responsible AI.
        * "full_response": a JSON value containing the response from the projects.locations.endpoints.generateContent call to the model.
        The generated text is in the text element.
        * "status": a STRING value that contains the API response status for the corresponding row. This value is empty if the operation was successful.
    """

    prompt_context, series_list = _separate_context_and_series(prompt)
    assert len(series_list) > 0

    operator = ai_ops.AIGenerateBool(
        prompt_context=tuple(prompt_context),
        connection_id=_resolve_connection_id(series_list[0], connection_id),
        endpoint=endpoint,
        request_type=request_type,
        model_params=json.dumps(model_params) if model_params else None,
    )

    return series_list[0]._apply_nary_op(operator, series_list[1:])


@log_adapter.method_logger(custom_base_name="bigquery_ai")
def generate_int(
    prompt: PROMPT_TYPE,
    *,
    connection_id: str | None = None,
    endpoint: str | None = None,
    request_type: Literal["dedicated", "shared", "unspecified"] = "unspecified",
    model_params: Mapping[Any, Any] | None = None,
) -> series.Series:
    """
    Returns the AI analysis based on the prompt, which can be any combination of text and unstructured data.

    **Examples:**

        >>> import bigframes.pandas as bpd
        >>> import bigframes.bigquery as bbq
        >>> bpd.options.display.progress_bar = None
        >>> animal = bpd.Series(["Kangaroo", "Rabbit", "Spider"])
        >>> bbq.ai.generate_int(("How many legs does a ", animal, " have?"))
        0    {'result': 2, 'full_response': '{"candidates":...
        1    {'result': 4, 'full_response': '{"candidates":...
        2    {'result': 8, 'full_response': '{"candidates":...
        dtype: struct<result: int64, full_response: extension<dbjson<JSONArrowType>>, status: string>[pyarrow]

        >>> bbq.ai.generate_int(("How many legs does a ", animal, " have?")).struct.field("result")
        0    2
        1    4
        2    8
        Name: result, dtype: Int64

    Args:
        prompt (Series | List[str|Series] | Tuple[str|Series, ...]):
            A mixture of Series and string literals that specifies the prompt to send to the model. The Series can be BigFrames Series
            or pandas Series.
        connection_id (str, optional):
            Specifies the connection to use to communicate with the model. For example, `myproject.us.myconnection`.
            If not provided, the connection from the current session will be used.
        endpoint (str, optional):
            Specifies the Vertex AI endpoint to use for the model. For example `"gemini-2.5-flash"`. You can specify any
            generally available or preview Gemini model. If you specify the model name, BigQuery ML automatically identifies and
            uses the full endpoint of the model. If you don't specify an ENDPOINT value, BigQuery ML selects a recent stable
            version of Gemini to use.
        request_type (Literal["dedicated", "shared", "unspecified"]):
            Specifies the type of inference request to send to the Gemini model. The request type determines what quota the request uses.
            * "dedicated": function only uses Provisioned Throughput quota. The function returns the error Provisioned throughput is not
            purchased or is not active if Provisioned Throughput quota isn't available.
            * "shared": the function only uses dynamic shared quota (DSQ), even if you have purchased Provisioned Throughput quota.
            * "unspecified": If you haven't purchased Provisioned Throughput quota, the function uses DSQ quota.
            If you have purchased Provisioned Throughput quota, the function uses the Provisioned Throughput quota first.
            If requests exceed the Provisioned Throughput quota, the overflow traffic uses DSQ quota.
        model_params (Mapping[Any, Any]):
            Provides additional parameters to the model. The MODEL_PARAMS value must conform to the generateContent request body format.

    Returns:
        bigframes.series.Series: A new struct Series with the result data. The struct contains these fields:
        * "result": an integer (INT64) value containing the model's response to the prompt. The result is None if the request fails or is filtered by responsible AI.
        * "full_response": a JSON value containing the response from the projects.locations.endpoints.generateContent call to the model.
        The generated text is in the text element.
        * "status": a STRING value that contains the API response status for the corresponding row. This value is empty if the operation was successful.
    """

    prompt_context, series_list = _separate_context_and_series(prompt)
    assert len(series_list) > 0

    operator = ai_ops.AIGenerateInt(
        prompt_context=tuple(prompt_context),
        connection_id=_resolve_connection_id(series_list[0], connection_id),
        endpoint=endpoint,
        request_type=request_type,
        model_params=json.dumps(model_params) if model_params else None,
    )

    return series_list[0]._apply_nary_op(operator, series_list[1:])


def _separate_context_and_series(
    prompt: PROMPT_TYPE,
) -> Tuple[List[str | None], List[series.Series]]:
    """
    Returns the two values. The first value is the prompt with all series replaced by None. The second value is all the series
    in the prompt. The original item order is kept.
    For example:
    Input: ("str1", series1, "str2", "str3", series2)
    Output: ["str1", None, "str2", "str3", None], [series1, series2]
    """
    if not isinstance(prompt, (list, tuple, series.Series)):
        raise ValueError(f"Unsupported prompt type: {type(prompt)}")

    if isinstance(prompt, series.Series):
        if prompt.dtype == dtypes.OBJ_REF_DTYPE:
            # Multi-model support
            return [None], [prompt.blob.read_url()]
        return [None], [prompt]

    prompt_context: List[str | None] = []
    series_list: List[series.Series | pd.Series] = []

    session = None
    for item in prompt:
        if isinstance(item, str):
            prompt_context.append(item)

        elif isinstance(item, (series.Series, pd.Series)):
            prompt_context.append(None)

            if isinstance(item, series.Series) and session is None:
                # Use the first available BF session if there's any.
                session = item._session
            series_list.append(item)

        else:
            raise TypeError(f"Unsupported type in prompt: {type(item)}")

    if not series_list:
        raise ValueError("Please provide at least one Series in the prompt")

    converted_list = [_convert_series(s, session) for s in series_list]

    return prompt_context, converted_list


def _convert_series(
    s: series.Series | pd.Series, session: session.Session | None
) -> series.Series:
    result = convert.to_bf_series(s, default_index=None, session=session)

    if result.dtype == dtypes.OBJ_REF_DTYPE:
        # Support multimodel
        return result.blob.read_url()
    return result


def _resolve_connection_id(series: series.Series, connection_id: str | None):
    return clients.get_canonical_bq_connection_id(
        connection_id or series._session._bq_connection,
        series._session._project,
        series._session._location,
    )
