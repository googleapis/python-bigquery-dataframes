# Contains code from https://github.com/ibis-project/ibis/expr/operations/udf.py

from __future__ import annotations

import enum


@enum.unique
class InputType(enum.Enum):
    BUILTIN = enum.auto()
    PANDAS = enum.auto()
    PYARROW = enum.auto()
    PYTHON = enum.auto()
