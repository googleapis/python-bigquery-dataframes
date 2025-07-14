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

from bigframes_vendored.ibis.expr import types as ibis_types

from bigframes.core.compile.ibis_compiler import scalar_op_compiler
from bigframes.operations.generic_ops import isnull_op


@scalar_op_compiler.scalar_op_compiler.register_unary_op(isnull_op.isnull_op)
def _ibis_isnull_op_impl(x: ibis_types.Value):
    return x.isnull()
