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

from typing import Dict, List, Union


def encode_value(v: Union[str, int, float, List[str]]) -> str:
    if isinstance(v, str):
        return f'"{v}"'
    elif isinstance(v, int) or isinstance(v, float):
        return f"{v}"
    elif isinstance(v, list):
        inner = ", ".join([encode_value(x) for x in v])
        return f"[{inner}]"
    else:
        raise ValueError("Unexpected value type")


def build_paramlist(indent: int, **kwargs) -> str:
    indent_str = "  " * indent
    param_strs = [f"{k}={encode_value(v)}" for k, v in kwargs.items()]
    return indent_str + f",\n{indent_str}".join(param_strs)


def create_model(
    model_name: str,
    source_sql: str,
    options: Dict[str, Union[str, int, float, List[str]]],
) -> str:
    # TODO(bmil): This should be CREATE TEMP MODEL after b/145824779 is fixed
    return f"""CREATE MODEL `{model_name}`
OPTIONS (
{build_paramlist(indent=1, **options)}
) AS {source_sql}"""


def ml_evaluate(model_name: str, source_sql: Union[str, None] = None) -> str:
    if source_sql is None:
        return f"""SELECT * FROM ML.EVALUATE(MODEL `{model_name}`)"""
    else:
        return f"""SELECT * FROM ML.EVALUATE(MODEL `{model_name}`,
  ({source_sql}))"""


def ml_predict(model_name: str, source_sql: str) -> str:
    return f"""SELECT * FROM ML.PREDICT(MODEL `{model_name}`,
  ({source_sql}))"""
