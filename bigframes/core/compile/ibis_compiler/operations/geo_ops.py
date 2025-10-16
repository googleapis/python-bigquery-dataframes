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

"""
BigFrames -> Ibis compilation for the operations in bigframes.operations.geo_ops.

Please keep implementations in sequential order by op name.
"""

from __future__ import annotations

from bigframes_vendored.ibis.expr import types as ibis_types
import bigframes_vendored.ibis.udf.scalar as ibis_udf

from bigframes.core.compile.ibis_compiler.scalar_op_compiler import scalar_op_compiler
from bigframes.operations import geo_ops

register_unary_op = scalar_op_compiler.register_unary_op
register_binary_op = scalar_op_compiler.register_binary_op


@ibis_udf.scalar.builtin("ST_IsEmpty")
def st_isempty(x: ibis_types.GeoValue) -> ibis_types.BooleanValue:
    ...


@register_unary_op(geo_ops.geo_st_isempty_op)
def geo_st_isempty_op_impl(x: ibis_types.Value):
    return st_isempty(x)


@ibis_udf.scalar.builtin("ST_GeometryType")
def st_geometrytype(x: ibis_types.GeoValue) -> ibis_types.StringValue:
    ...


@register_unary_op(geo_ops.geo_st_geometrytype_op)
def geo_st_geometrytype_op_impl(x: ibis_types.Value):
    return st_geometrytype(x)


@ibis_udf.scalar.builtin("ST_IsRing")
def st_isring(x: ibis_types.GeoValue) -> ibis_types.BooleanValue:
    ...


@register_unary_op(geo_ops.geo_st_isring_op)
def geo_st_isring_op_impl(x: ibis_types.Value):
    return st_isring(x)


@ibis_udf.scalar.builtin("ST_EQUALS")
def st_equals(
    x: ibis_types.GeoValue, y: ibis_types.GeoValue
) -> ibis_types.BooleanValue:
    ...


@ibis_udf.scalar.builtin("ST_SIMPLIFY")
def st_simplify(
    x: ibis_types.GeoValue, tolerance: ibis_types.NumericValue
) -> ibis_types.GeoValue:
    ...


@register_unary_op(geo_ops.geo_st_issimple_op)
def geo_st_issimple_op_impl(x: ibis_types.Value):
    simplified = st_simplify(x, 0.0)
    return st_equals(x, simplified)


@ibis_udf.scalar.builtin("ST_ISVALID")
def st_isvalid(x: ibis_types.GeoValue) -> ibis_types.BooleanValue:
    ...


@register_unary_op(geo_ops.geo_st_isvalid_op)
def geo_st_isvalid_op_impl(x: ibis_types.Value):
    return st_isvalid(x)


@ibis_udf.scalar.builtin("ST_UNION")
def st_union(
    x: ibis_types.GeoValue, y: ibis_types.GeoValue
) -> ibis_types.GeoValue:
    ...


@register_binary_op(geo_ops.geo_st_union_op)
def geo_st_union_op_impl(
    x: ibis_types.Value, y: ibis_types.Value
) -> ibis_types.Value:
    return st_union(x, y)