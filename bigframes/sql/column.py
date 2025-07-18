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

import abc
import dataclasses

from bigframes.core import expression
import bigframes.core.guid
import bigframes.operations as ops


@dataclasses.dataclass
class Column(abc.ABC):
    _value: expression.Expression
    _alias: str = dataclasses.field(default_factory=bigframes.core.guid.generate_guid)

    def _to_bf_expr(self) -> expression.Expression:
        return self._value

    def __add__(self, other) -> Column:
        if not isinstance(other, Column):
            other = Column(expression.const(other))
        return Column(ops.add_op.as_expr(self._value, other._value))

    def alias(self, name) -> Column:
        return Column(self._value, _alias=name)
