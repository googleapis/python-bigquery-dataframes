# Copyright 2024 Google LLC
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

import dataclasses
import typing

import bigframes.core.transpiler.googlesql.expression as expr


@dataclasses.dataclass
class DataType:
    """The base class of GoogleSQL for BigQuery data types"""

    def sql(self):
        return self.__class__.__name__.upper()


@dataclasses.dataclass
class Int64(DataType):
    pass


@dataclasses.dataclass
class Float64(DataType):
    pass


@dataclasses.dataclass
class Numeric(DataType):
    pass


@dataclasses.dataclass
class Bignumeric(DataType):
    pass


@dataclasses.dataclass
class Bool(DataType):
    pass


@dataclasses.dataclass
class String(DataType):
    pass


@dataclasses.dataclass
class Array(DataType):
    type: DataType

    def sql(self):
        return f"ARRAY<{self.type.sql()}>"


@dataclasses.dataclass
class Struct(DataType):
    exprs: typing.Sequence[expr.ABCExpression]
    types: typing.Sequence[DataType]

    def __post_init__(self):
        if len(self.exprs) != len(self.types):
            raise ValueError(
                f"The length of expressions({len(self.exprs)}) and "
                + f"types({len(self.types)}) should be same."
            )

    def sql(self):
        exprs_list = [
            f"{expr.sql()} {type.sql()}" for expr, type in zip(self.exprs, self.types)
        ]
        return f"STRUCT<{', '.join(exprs_list)}>"
