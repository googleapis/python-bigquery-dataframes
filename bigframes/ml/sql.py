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

"""
Generates SQL queries needed for BigFrames ML
"""

from typing import List, Optional, Union


def encode_value(v: Union[str, int, float, List[str]]) -> str:
    """Encode a parameter value for SQL"""
    if isinstance(v, str):
        return f'"{v}"'
    elif isinstance(v, int) or isinstance(v, float):
        return f"{v}"
    elif isinstance(v, list):
        inner = ", ".join([encode_value(x) for x in v])
        return f"[{inner}]"
    else:
        raise ValueError("Unexpected value type")


def build_param_list(**kwargs: Union[str, int, float, List[str]]) -> str:
    """Encode a dict of values into a formatted list of KVPs for SQL"""
    indent_str = "  "
    param_strs = [f"{k}={encode_value(v)}" for k, v in kwargs.items()]
    return "\n" + indent_str + f",\n{indent_str}".join(param_strs)


def options(**kwargs: Union[str, int, float, List[str]]) -> str:
    """Encode the OPTIONS clause for BQML"""
    return f"OPTIONS({build_param_list(**kwargs)})"


def build_expr_list(*expr_sqls: str) -> str:
    "Encode a list of SQL expressions into a formatted list for SQL"
    indent_str = "  "
    return "\n" + indent_str + f",\n{indent_str}".join(expr_sqls)


def transform(*expr_sqls: str) -> str:
    """Encode the TRANSFORM clause for BQML"""
    return f"TRANSFORM({build_expr_list(*expr_sqls)})"


def connection(conn_name: str) -> str:
    """Encode the REMOTE WITH CONNECTION clause for BQML. conn_name is of the format <PROJECT_NUMBER/PROJECT_ID>.<REGION>.<CONNECTION_NAME>."""
    return f"REMOTE WITH CONNECTION `{conn_name}`"


def ml_standard_scaler(numeric_expr_sql: str, name: str) -> str:
    """Encode ML.STANDARD_SCALER for BQML"""
    return f"""ML.STANDARD_SCALER({numeric_expr_sql}) OVER() AS {name}"""


def ml_one_hot_encoder(numeric_expr_sql: str, name: str) -> str:
    """Encode ML.ONE_HOT_ENCODER for BQML"""
    return f"""ML.ONE_HOT_ENCODER({numeric_expr_sql}) OVER() AS {name}"""


def create_model(
    model_name: str,
    source_sql: str,
    transform_sql: Optional[str] = None,
    options_sql: Optional[str] = None,
) -> str:
    """Encode the CREATE MODEL statement for BQML"""
    # TODO(bmil): This should be CREATE TEMP MODEL after b/145824779 is fixed
    parts = [f"CREATE MODEL `{model_name}`"]
    if transform_sql:
        parts.append(transform_sql)
    if options_sql:
        parts.append(options_sql)
    parts.append(f"AS {source_sql}")
    return "\n".join(parts)


def create_remote_model(
    model_name: str,
    connection_name: str,
    options_sql: Optional[str] = None,
) -> str:
    """Encode the CREATE MODEL statement for BQML"""
    # TODO(bmil): This should be CREATE TEMP MODEL after b/145824779 is fixed
    parts = [f"CREATE MODEL `{model_name}`"]
    parts.append(connection(connection_name))
    if options_sql:
        parts.append(options_sql)
    return "\n".join(parts)


def ml_evaluate(model_name: str, source_sql: Union[str, None] = None) -> str:
    """Encode ML.EVALUATE for BQML"""
    if source_sql is None:
        return f"""SELECT * FROM ML.EVALUATE(MODEL `{model_name}`)"""
    else:
        return f"""SELECT * FROM ML.EVALUATE(MODEL `{model_name}`,
  ({source_sql}))"""


def ml_predict(model_name: str, source_sql: str) -> str:
    """Encode ML.PREDICT for BQML"""
    return f"""SELECT * FROM ML.PREDICT(MODEL `{model_name}`,
  ({source_sql}))"""


def ml_transform(model_name: str, source_sql: str) -> str:
    """Encode ML.TRANSFORM for BQML"""
    return f"""SELECT * FROM ML.TRANSFORM(MODEL `{model_name}`,
  ({source_sql}))"""


def ml_generate_text(model_name: str, source_sql: str) -> str:
    """Encode ML.GENERATE_TEXT for BQML"""
    return f"""SELECT * FROM ML.GENERATE_TEXT(MODEL `{model_name}`,
  ({source_sql}))"""
