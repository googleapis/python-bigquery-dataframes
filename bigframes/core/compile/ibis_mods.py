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


from typing import Callable, Type

from ibis.backends.bigquery.registry import OPERATION_REGISTRY
from ibis.expr.operations.core import Binary, Unary

import bigframes.dtypes as bf_dt
import bigframes.operations as ops


def create_unary_op(op: type[ops.RowOp], write_rule: Callable[..., str]) -> Type[Unary]:
    class CustomUnaryOp(Unary):
        @property
        def dtype(self):
            input_bigframes_types = [
                bf_dt.ibis_dtype_to_bigframes_dtype(self.arg.dtype)
            ]
            output_bigframes_type = bf_dt.etype_to_dtype(
                op().output_type(*input_bigframes_types)
            )
            return bf_dt.bigframes_dtype_to_ibis_dtype(output_bigframes_type)

    def _impl(translator, op: CustomUnaryOp):
        return write_rule(translator.translate(op.arg))

    OPERATION_REGISTRY.update({CustomUnaryOp: _impl})
    return CustomUnaryOp


def create_binary_op(
    op: type[ops.RowOp], write_rule: Callable[..., str]
) -> Type[Binary]:
    class CustomBinaryOp(Binary):
        @property
        def dtype(self):
            input_bigframes_types = map(
                lambda arg: bf_dt.ibis_dtype_to_bigframes_dtype(arg.dtype),
                [self.left, self.right],
            )
            output_bigframes_type = bf_dt.etype_to_dtype(
                op().output_type(*input_bigframes_types)
            )
            return bf_dt.bigframes_dtype_to_ibis_dtype(output_bigframes_type)

    def _impl(translator, op: CustomBinaryOp):
        return write_rule(translator.translate(op.left), translator.translate(op.right))

    OPERATION_REGISTRY.update({CustomBinaryOp: _impl})
    return CustomBinaryOp
