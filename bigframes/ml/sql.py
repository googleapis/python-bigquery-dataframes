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


def create_replace_model(
    model_name: str,
    source_sql: str,
    options: Dict[str, Union[str, int, float, List[str]]],
) -> str:
    return f"""CREATE OR REPLACE MODEL `{model_name}`
OPTIONS (
{build_paramlist(indent=1, **options)}
) AS {source_sql}"""


def ml_evaluate(model_name: str, source_sql: Union[str, None] = None) -> str:
    if source_sql is None:
        return f"""ML.EVALUATE(model_name="{model_name}")"""
    else:
        return f"""ML.EVALUATE(model_name="{model_name}",
  ({source_sql}))"""


def ml_predict(model_name: str, source_sql: str) -> str:
    return f"""ML.PREDICT(model_name="{model_name}",
  ({source_sql}))"""
