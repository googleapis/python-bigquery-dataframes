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

import datetime
import typing

import ibis.expr.types as ibis_types

ImmediateScalar = typing.Union[
    bool, int, float, str, datetime.date, datetime.date, datetime.datetime
]


class DeferredScalar:
    """A deferred scalar object."""

    def __init__(self, value: ibis_types.Scalar):
        self._value = value

    def __repr__(self) -> str:
        """Converts a Series to a string."""
        # TODO(swast): Add a timeout here? If the query is taking a long time,
        # maybe we just print the job metadata that we have so far?
        return repr(self.compute())

    def compute(self) -> ImmediateScalar:
        """Executes deferred operations and downloads the resulting scalar."""
        return self._value.execute()


# All public APIs return ImmediateScalar at present
# Later implementation may sometimes return a lazy scalar
Scalar = ImmediateScalar
