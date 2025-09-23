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

import dataclasses
from typing import ClassVar, Literal, Tuple

import pandas as pd
import pyarrow as pa

from bigframes import dtypes
from bigframes.operations import base_ops


@dataclasses.dataclass(frozen=True)
class AIGenerateBool(base_ops.NaryOp):
    name: ClassVar[str] = "ai_generate_bool"

    prompt_context: Tuple[str | None, ...]
    connection_id: str
    endpoint: str | None
    request_type: Literal["dedicated", "shared", "unspecified"]
    model_params: str | None

    def output_type(self, *input_types: dtypes.ExpressionType) -> dtypes.ExpressionType:
        return pd.ArrowDtype(
            pa.struct(
                (
                    pa.field("result", pa.bool_()),
                    pa.field("full_response", dtypes.JSON_ARROW_TYPE),
                    pa.field("status", pa.string()),
                )
            )
        )


@dataclasses.dataclass(frozen=True)
class AIGenerateInt(base_ops.NaryOp):
    name: ClassVar[str] = "ai_generate_int"

    prompt_context: Tuple[str | None, ...]
    connection_id: str
    endpoint: str | None
    request_type: Literal["dedicated", "shared", "unspecified"]
    model_params: str | None

    def output_type(self, *input_types: dtypes.ExpressionType) -> dtypes.ExpressionType:
        return pd.ArrowDtype(
            pa.struct(
                (
                    pa.field("result", pa.int64()),
                    pa.field("full_response", dtypes.JSON_ARROW_TYPE),
                    pa.field("status", pa.string()),
                )
            )
        )
