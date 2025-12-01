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

import typing
from typing import Mapping, Optional, Union

import bigframes.core.compile.googlesql as googlesql
import bigframes.core.sql

def create_model_ddl(
    model_name: str,
    *,
    replace: bool = False,
    if_not_exists: bool = False,
    transform: Optional[list[str]] = None,
    input_schema: Optional[Mapping[str, str]] = None,
    output_schema: Optional[Mapping[str, str]] = None,
    connection_name: Optional[str] = None,
    options: Optional[Mapping[str, Union[str, int, float, bool, list]]] = None,
    training_data: Optional[str] = None,
    custom_holiday: Optional[str] = None,
) -> str:
    """Encode the CREATE MODEL statement.

    See https://docs.cloud.google.com/bigquery/docs/reference/standard-sql/bigqueryml-syntax-create for reference.
    """

    if replace:
        create = "CREATE OR REPLACE MODEL "
    elif if_not_exists:
        create = "CREATE MODEL IF NOT EXISTS "
    else:
        create = "CREATE MODEL "

    ddl = f"{create}{googlesql.identifier(model_name)}\n"

    # [TRANSFORM (select_list)]
    if transform:
        ddl += f"TRANSFORM ({', '.join(transform)})\n"

    # [INPUT (field_name field_type) OUTPUT (field_name field_type)]
    if input_schema:
        inputs = [f"{k} {v}" for k, v in input_schema.items()]
        ddl += f"INPUT ({', '.join(inputs)})\n"

    if output_schema:
        outputs = [f"{k} {v}" for k, v in output_schema.items()]
        ddl += f"OUTPUT ({', '.join(outputs)})\n"

    # [REMOTE WITH CONNECTION {connection_name | DEFAULT}]
    if connection_name:
        if connection_name.upper() == "DEFAULT":
             ddl += "REMOTE WITH CONNECTION DEFAULT\n"
        else:
             ddl += f"REMOTE WITH CONNECTION {googlesql.identifier(connection_name)}\n"

    # [OPTIONS(model_option_list)]
    if options:
        rendered_options = []
        for option_name, option_value in options.items():
            if isinstance(option_value, (list, tuple)):
                # Handle list options like model_registry="vertex_ai"
                # wait, usually options are key=value.
                # if value is list, it is [val1, val2]
                rendered_val = bigframes.core.sql.simple_literal(list(option_value))
            else:
                 rendered_val = bigframes.core.sql.simple_literal(option_value)

            rendered_options.append(f"{option_name} = {rendered_val}")

        ddl += f"OPTIONS({', '.join(rendered_options)})\n"

    # [AS {query_statement | ( training_data AS (query_statement), custom_holiday AS (holiday_statement) )}]

    if training_data:
        if custom_holiday:
            # When custom_holiday is present, we need named clauses
            parts = []
            parts.append(f"training_data AS ({training_data})")
            parts.append(f"custom_holiday AS ({custom_holiday})")
            ddl += f"AS (\n  {', '.join(parts)}\n)"
        else:
             # Just training_data is treated as the query_statement
             ddl += f"AS {training_data}"

    return ddl
