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

import dataclasses
import typing

from bigframes import dtypes
from bigframes.operations import base_ops
import bigframes.operations.type as op_typing

sin_op = base_ops.create_unary_op(
    name="sin", type_signature=op_typing.UNARY_REAL_NUMERIC
)

cos_op = base_ops.create_unary_op(
    name="cos", type_signature=op_typing.UNARY_REAL_NUMERIC
)

tan_op = base_ops.create_unary_op(
    name="tan", type_signature=op_typing.UNARY_REAL_NUMERIC
)

arcsin_op = base_ops.create_unary_op(
    name="arcsin", type_signature=op_typing.UNARY_REAL_NUMERIC
)

arccos_op = base_ops.create_unary_op(
    name="arccos", type_signature=op_typing.UNARY_REAL_NUMERIC
)

arctan_op = base_ops.create_unary_op(
    name="arctan", type_signature=op_typing.UNARY_REAL_NUMERIC
)

sinh_op = base_ops.create_unary_op(
    name="sinh", type_signature=op_typing.UNARY_REAL_NUMERIC
)

cosh_op = base_ops.create_unary_op(
    name="cosh", type_signature=op_typing.UNARY_REAL_NUMERIC
)

tanh_op = base_ops.create_unary_op(
    name="tanh", type_signature=op_typing.UNARY_REAL_NUMERIC
)

arcsinh_op = base_ops.create_unary_op(
    name="arcsinh", type_signature=op_typing.UNARY_REAL_NUMERIC
)

arccosh_op = base_ops.create_unary_op(
    name="arccosh", type_signature=op_typing.UNARY_REAL_NUMERIC
)

arctanh_op = base_ops.create_unary_op(
    name="arctanh", type_signature=op_typing.UNARY_REAL_NUMERIC
)

floor_op = base_ops.create_unary_op(
    name="floor", type_signature=op_typing.UNARY_REAL_NUMERIC
)

ceil_op = base_ops.create_unary_op(
    name="ceil", type_signature=op_typing.UNARY_REAL_NUMERIC
)

abs_op = base_ops.create_unary_op(
    name="abs", type_signature=op_typing.UNARY_NUMERIC_AND_TIMEDELTA
)

pos_op = base_ops.create_unary_op(
    name="pos", type_signature=op_typing.UNARY_NUMERIC_AND_TIMEDELTA
)

neg_op = base_ops.create_unary_op(
    name="neg", type_signature=op_typing.UNARY_NUMERIC_AND_TIMEDELTA
)

exp_op = base_ops.create_unary_op(
    name="exp", type_signature=op_typing.UNARY_REAL_NUMERIC
)

expm1_op = base_ops.create_unary_op(
    name="expm1", type_signature=op_typing.UNARY_REAL_NUMERIC
)

ln_op = base_ops.create_unary_op(
    name="log", type_signature=op_typing.UNARY_REAL_NUMERIC
)

log10_op = base_ops.create_unary_op(
    name="log10", type_signature=op_typing.UNARY_REAL_NUMERIC
)

log1p_op = base_ops.create_unary_op(
    name="log1p", type_signature=op_typing.UNARY_REAL_NUMERIC
)

sqrt_op = base_ops.create_unary_op(
    name="sqrt", type_signature=op_typing.UNARY_REAL_NUMERIC
)


@dataclasses.dataclass(frozen=True)
class AddOp(base_ops.BinaryOp):
    name: typing.ClassVar[str] = "add"

    def output_type(self, *input_types):
        left_type = input_types[0]
        right_type = input_types[1]
        if all(map(dtypes.is_string_like, input_types)) and len(set(input_types)) == 1:
            # String addition
            return input_types[0]

        # Timestamp addition.
        if dtypes.is_datetime_like(left_type) and right_type is dtypes.TIMEDELTA_DTYPE:
            return left_type
        if left_type is dtypes.TIMEDELTA_DTYPE and dtypes.is_datetime_like(right_type):
            return right_type

        if left_type is dtypes.TIMEDELTA_DTYPE and right_type is dtypes.TIMEDELTA_DTYPE:
            return dtypes.TIMEDELTA_DTYPE

        if (left_type is None or dtypes.is_numeric(left_type)) and (
            right_type is None or dtypes.is_numeric(right_type)
        ):
            # Numeric addition
            return dtypes.coerce_to_common(left_type, right_type)
        raise TypeError(f"Cannot add dtypes {left_type} and {right_type}")


add_op = AddOp()


@dataclasses.dataclass(frozen=True)
class SubOp(base_ops.BinaryOp):
    name: typing.ClassVar[str] = "sub"

    # Note: this is actualyl a vararg op, but we don't model that yet
    def output_type(self, *input_types):
        left_type = input_types[0]
        right_type = input_types[1]

        if dtypes.is_datetime_like(left_type) and dtypes.is_datetime_like(right_type):
            return dtypes.TIMEDELTA_DTYPE

        if dtypes.is_datetime_like(left_type) and right_type is dtypes.TIMEDELTA_DTYPE:
            return left_type

        if left_type is dtypes.TIMEDELTA_DTYPE and right_type is dtypes.TIMEDELTA_DTYPE:
            return dtypes.TIMEDELTA_DTYPE

        if (left_type is None or dtypes.is_numeric(left_type)) and (
            right_type is None or dtypes.is_numeric(right_type)
        ):
            # Numeric subtraction
            return dtypes.coerce_to_common(left_type, right_type)

        raise TypeError(f"Cannot subtract dtypes {left_type} and {right_type}")


sub_op = SubOp()


@dataclasses.dataclass(frozen=True)
class MulOp(base_ops.BinaryOp):
    name: typing.ClassVar[str] = "mul"

    def output_type(self, *input_types: dtypes.ExpressionType) -> dtypes.ExpressionType:
        left_type = input_types[0]
        right_type = input_types[1]

        if left_type is dtypes.TIMEDELTA_DTYPE and dtypes.is_numeric(right_type):
            return dtypes.TIMEDELTA_DTYPE
        if dtypes.is_numeric(left_type) and right_type is dtypes.TIMEDELTA_DTYPE:
            return dtypes.TIMEDELTA_DTYPE

        if (left_type is None or dtypes.is_numeric(left_type)) and (
            right_type is None or dtypes.is_numeric(right_type)
        ):
            return dtypes.coerce_to_common(left_type, right_type)

        raise TypeError(f"Cannot multiply dtypes {left_type} and {right_type}")


mul_op = MulOp()


@dataclasses.dataclass(frozen=True)
class DivOp(base_ops.BinaryOp):
    name: typing.ClassVar[str] = "div"

    def output_type(self, *input_types: dtypes.ExpressionType) -> dtypes.ExpressionType:
        left_type = input_types[0]
        right_type = input_types[1]

        if left_type is dtypes.TIMEDELTA_DTYPE and dtypes.is_numeric(right_type):
            return dtypes.TIMEDELTA_DTYPE

        if left_type is dtypes.TIMEDELTA_DTYPE and right_type is dtypes.TIMEDELTA_DTYPE:
            return dtypes.FLOAT_DTYPE

        if (left_type is None or dtypes.is_numeric(left_type)) and (
            right_type is None or dtypes.is_numeric(right_type)
        ):
            lcd_type = dtypes.coerce_to_common(left_type, right_type)
            # Real numeric ops produce floats on int input
            return dtypes.FLOAT_DTYPE if lcd_type == dtypes.INT_DTYPE else lcd_type

        raise TypeError(f"Cannot divide dtypes {left_type} and {right_type}")


div_op = DivOp()


@dataclasses.dataclass(frozen=True)
class FloorDivOp(base_ops.BinaryOp):
    name: typing.ClassVar[str] = "floordiv"

    def output_type(self, *input_types: dtypes.ExpressionType) -> dtypes.ExpressionType:
        left_type = input_types[0]
        right_type = input_types[1]

        if left_type is dtypes.TIMEDELTA_DTYPE and dtypes.is_numeric(right_type):
            return dtypes.TIMEDELTA_DTYPE

        if left_type is dtypes.TIMEDELTA_DTYPE and right_type is dtypes.TIMEDELTA_DTYPE:
            return dtypes.INT_DTYPE

        if (left_type is None or dtypes.is_numeric(left_type)) and (
            right_type is None or dtypes.is_numeric(right_type)
        ):
            return dtypes.coerce_to_common(left_type, right_type)

        raise TypeError(f"Cannot floor divide dtypes {left_type} and {right_type}")


floordiv_op = FloorDivOp()

pow_op = base_ops.create_binary_op(name="pow", type_signature=op_typing.BINARY_NUMERIC)

mod_op = base_ops.create_binary_op(name="mod", type_signature=op_typing.BINARY_NUMERIC)

arctan2_op = base_ops.create_binary_op(
    name="arctan2", type_signature=op_typing.BINARY_REAL_NUMERIC
)

round_op = base_ops.create_binary_op(
    name="round", type_signature=op_typing.BINARY_REAL_NUMERIC
)

unsafe_pow_op = base_ops.create_binary_op(
    name="unsafe_pow_op", type_signature=op_typing.BINARY_REAL_NUMERIC
)
