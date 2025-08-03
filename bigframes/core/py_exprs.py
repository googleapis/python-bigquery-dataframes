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

from __future__ import annotations

import dataclasses
import itertools
from types import ModuleType
from typing import Callable, Mapping, Tuple

from bigframes import dtypes
from bigframes.core import identifiers
from bigframes.core.expression import Expression, OpExpression
from bigframes.operations import NUMPY_TO_BINOP, NUMPY_TO_OP

_CALLABLE_TO_OP = {
    **NUMPY_TO_OP,
    **NUMPY_TO_BINOP,
}


@dataclasses.dataclass(frozen=True)
class GetAttr(Expression):
    input: Expression
    attr: str

    @property
    def column_references(
        self,
    ) -> Tuple[identifiers.ColumnId, ...]:
        return self.input.column_references

    @property
    def free_variables(self) -> set[str]:
        return self.input.free_variables

    @property
    def is_const(self) -> bool:
        return False

    @property
    def children(self):
        return (self.input,)

    @property
    def nullable(self) -> bool:
        return True

    @property
    def is_resolved(self) -> bool:
        return False

    @property
    def output_type(self) -> dtypes.ExpressionType:
        raise ValueError(f"Type of expression {self} has not been fixed.")

    @property
    def is_bijective(self) -> bool:
        # TODO: Mark individual functions as bijective?
        return False

    @property
    def deterministic(self) -> bool:
        return True

    def transform_children(self, t: Callable[[Expression], Expression]) -> Expression:
        new_input = t(self.input)
        if new_input != self.input:
            return dataclasses.replace(self, input=new_input)
        return self

    def bind_variables(
        self, bindings: Mapping[str, Expression], allow_partial_bindings: bool = False
    ) -> GetAttr:
        return GetAttr(
            self.input.bind_variables(
                bindings, allow_partial_bindings=allow_partial_bindings
            ),
            self.attr,
        )

    def bind_refs(
        self,
        bindings: Mapping[identifiers.ColumnId, Expression],
        allow_partial_bindings: bool = False,
    ) -> GetAttr:
        return GetAttr(
            self.input.bind_refs(
                bindings, allow_partial_bindings=allow_partial_bindings
            ),
            self.attr,
        )


@dataclasses.dataclass(frozen=True)
class Module(Expression):
    """An expression representing a module reference."""

    module: ModuleType

    @property
    def is_const(self) -> bool:
        return True

    @property
    def column_references(self) -> Tuple[identifiers.ColumnId, ...]:
        return ()

    @property
    def nullable(self) -> bool:
        return True  # type: ignore

    @property
    def is_resolved(self) -> bool:
        return False

    @property
    def output_type(self) -> dtypes.ExpressionType:
        raise ValueError("Module expresion has not type")

    def bind_variables(
        self, bindings: Mapping[str, Expression], allow_partial_bindings: bool = False
    ) -> Expression:
        return self

    def bind_refs(
        self,
        bindings: Mapping[identifiers.ColumnId, Expression],
        allow_partial_bindings: bool = False,
    ) -> Module:
        return self

    @property
    def is_bijective(self) -> bool:
        # () <-> value
        return True

    def transform_children(self, t: Callable[[Expression], Expression]) -> Expression:
        return self


@dataclasses.dataclass(frozen=True)
class Call(Expression):
    """An expression representing a scalar constant."""

    # TODO: Further constrain?
    callable: Expression
    inputs: Tuple[Expression, ...]

    @property
    def column_references(
        self,
    ) -> Tuple[identifiers.ColumnId, ...]:
        return tuple(
            itertools.chain.from_iterable(
                map(lambda x: x.column_references, self.children)
            )
        )

    @property
    def free_variables(self) -> set[str]:
        return set(
            itertools.chain.from_iterable(
                map(lambda x: x.free_variables, self.children)
            )
        )

    @property
    def is_const(self) -> bool:
        return False

    @property
    def children(self):
        return (self.callable, *self.inputs)

    @property
    def nullable(self) -> bool:
        return True

    @property
    def is_resolved(self) -> bool:
        return False

    @property
    def output_type(self) -> dtypes.ExpressionType:
        raise ValueError(f"Type of expression {self} has not been fixed.")

    @property
    def is_bijective(self) -> bool:
        # TODO: Mark individual functions as bijective?
        return False

    @property
    def deterministic(self) -> bool:
        return True

    def transform_children(self, t: Callable[[Expression], Expression]) -> Expression:
        return dataclasses.replace(
            self,
            callable=t(self.callable),
            inputs=tuple(t(input) for input in self.inputs),
        )

    def bind_variables(
        self, bindings: Mapping[str, Expression], allow_partial_bindings: bool = False
    ) -> Call:
        return Call(
            callable=self.callable.bind_variables(
                bindings, allow_partial_bindings=allow_partial_bindings
            ),
            inputs=tuple(
                input.bind_variables(
                    bindings, allow_partial_bindings=allow_partial_bindings
                )
                for input in self.inputs
            ),
        )

    def bind_refs(
        self,
        bindings: Mapping[identifiers.ColumnId, Expression],
        allow_partial_bindings: bool = False,
    ) -> Call:
        return Call(
            callable=self.callable.bind_refs(
                bindings, allow_partial_bindings=allow_partial_bindings
            ),
            inputs=tuple(
                input.bind_refs(bindings, allow_partial_bindings=allow_partial_bindings)
                for input in self.inputs
            ),
        )


def resolve_call(call: Call) -> Expression:
    callable = call.callable
    if isinstance(callable, GetAttr):
        attr = callable.attr
        if isinstance(callable.input, Module):
            fn = getattr(callable.input.module, attr)
            if fn in _CALLABLE_TO_OP:
                op = _CALLABLE_TO_OP[fn]
                return OpExpression(op, call.inputs)

    raise NotImplementedError(
        f"No implementation available for call expression: {call}"
    )
