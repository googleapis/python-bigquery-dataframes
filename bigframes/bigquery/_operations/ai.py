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

from __future__ import annotations

import json

from typing import List, Literal, Mapping, Sequence, Tuple, Any

from bigframes import series
from bigframes.operations import ai_ops


def ai_generate_bool(
    prompt: series.Series | Sequence[str | series.Series],
    *,
    connection_id: str | None = None,
    endpoint: str | None = None,
    request_type: Literal["dedicated", "shared", "unspecified"] = "unspecified",
    model_params: Mapping[Any, Any] | None = None,
) -> series.Series:
    """ """

    if request_type not in ("dedicated", "shared", "unspecified"):
        raise ValueError(f"Unsupported request type: {request_type}")

    if isinstance(prompt, series.Series):
        prompt_context, series_list = _separate_context_and_series([prompt])
    elif isinstance(prompt, Sequence):
        prompt_context, series_list = _separate_context_and_series(prompt)
    else:
        raise ValueError(f"Unsupported prompt type: {type(prompt)}")

    if not series_list:
        raise ValueError("Please provide at least one Series in the prompt")
    
    operator = ai_ops.AIGenerateBool(
        tuple(prompt_context),
        connection_id=connection_id or series_list[0]._session._bq_connection,
        endpoint=endpoint,
        request_type=request_type,
        model_params=json.dumps(model_params) if model_params else None,
    )

    return series_list[0]._apply_nary_op(operator, series_list[1:])


def _separate_context_and_series(
    prompt: Sequence[str | series.Series],
) -> Tuple[List[str | None], List[series.Series]]:
    """
    Returns the two values. The first value is the prompt with all series replaced by None. The second value is all the series
    in the prompt. The original item order is kept. 

    For example:
    Input: ("str1", series1, "str2", "str3", series2)
    Output: ["str1", None, "str2", "str3", None], [series1, series2]
    """

    prompt_context: List[str|None] = []
    series_list: List[series.Series] = []

    for item in prompt:
        if isinstance(item, str):
            prompt_context.append(item)
        elif isinstance(item, series.Series):
            prompt_context.append(None)
        else:
            raise ValueError(f"Unsupported type in prompt: {type(item)}")

    return prompt_context, series_list