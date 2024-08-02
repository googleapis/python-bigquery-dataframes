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

"""DataFrame is a two dimensional data structure."""

from __future__ import annotations

import functools
from typing import Optional, Protocol, TYPE_CHECKING

import bigframes.constants
import bigframes.exceptions

if TYPE_CHECKING:
    from bigframes import Session
    from bigframes.core.blocks import Block


class HasSession(Protocol):
    @property
    def _session(self) -> Session:
        ...

    @property
    def _block(self) -> Block:
        ...


def requires_ordering(suggestion: Optional[str] = None):
    def decorator(meth):
        @functools.wraps(meth)
        def guarded_meth(object: HasSession, *args, **kwargs):
            enforce_ordered(object, meth.__name__, suggestion)
            return meth(object, *args, **kwargs)

        return guarded_meth

    return decorator


def enforce_ordered(
    object: HasSession, opname: str, suggestion: Optional[str] = None
) -> None:
    session = object._session
    if session._strictly_ordered or not object._block.expr.node.order_ambiguous:
        # No ambiguity for how to calculate ordering, so no error or warning
        return None
    if not session._allows_ambiguity:
        suggestion_substr = suggestion + " " if suggestion else ""
        raise bigframes.exceptions.OrderRequiredError(
            f"Op {opname} not supported when strict ordering is disabled. {suggestion_substr}{bigframes.constants.FEEDBACK_LINK}"
        )
    if not object._block.explicitly_ordered:
        raise bigframes.exceptions.OrderRequiredError(
            f"Op {opname} requires an ordering. Use .sort_values or .sort_index to provide an ordering. {bigframes.constants.FEEDBACK_LINK}"
        )
