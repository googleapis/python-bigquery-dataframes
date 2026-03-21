# Copyright 2026 Google LLC
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

"""Compiler for BigFrames expression to Apache DataFusion expression.

Make sure to import all datafusion implementations here so that they get registered.
"""
from __future__ import annotations

import warnings

import bigframes.core.compile.datafusion.operations.comparison_ops  # noqa: F401

# The ops imports appear first so that the implementations can be registered.
import bigframes.core.compile.datafusion.operations.generic_ops  # noqa: F401
import bigframes.core.compile.datafusion.operations.numeric_ops  # noqa: F401

try:
    import bigframes._importing

    bigframes._importing.import_datafusion()

    from bigframes.core.compile.datafusion.compiler import DataFusionCompiler

    __all__ = ["DataFusionCompiler"]
except Exception as exc:
    msg = f"DataFusion compiler not available as there was an exception importing datafusion. Details: {str(exc)}"
    warnings.warn(msg)
