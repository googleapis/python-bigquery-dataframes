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

from bigframes_vendored import ibis
from bigframes_vendored.ibis.expr import types as ibis_types
import bigframes_vendored.ibis.expr.datatypes as ibis_dtypes
import bigframes_vendored.ibis.expr.operations.geospatial as ibis_geo

from bigframes.core.compile.ibis_compiler import scalar_op_compiler
from bigframes.operations import geo_ops

register_unary_op = scalar_op_compiler.scalar_op_compiler.register_unary_op


@register_unary_op(geo_ops.StRegionStatsOp, pass_op=True)
def st_regionstats(
    geography: ibis_types.Value,
    op: geo_ops.StRegionStatsOp,
):

    if op.band:
        band = ibis.literal(op.band, type=ibis_dtypes.string())
    else:
        band = None

    if op.include:
        include = ibis.literal(op.include, type=ibis_dtypes.string())
    else:
        include = None

    if op.options:
        options = ibis.literal(op.options, type=ibis_dtypes.json())
    else:
        options = None

    return ibis_geo.GeoRegionStats(
        arg=geography,
        raster_id=ibis.literal(op.raster_id, type=ibis_dtypes.string()),
        band=band,
        include=include,
        options=options,
    ).to_expr()
