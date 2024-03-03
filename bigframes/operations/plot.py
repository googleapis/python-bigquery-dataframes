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

from typing import Sequence

import bigframes.constants as constants
import bigframes.operations._matplotlib as plotbackend
import third_party.bigframes_vendored.pandas.plotting._core as vendordt


class PlotAccessor:
    __doc__ = vendordt.PlotAccessor.__doc__

    def __init__(self, data) -> None:
        self._parent = data

    def hist(self, by: Sequence[str] | None = None, bins: int = 10, **kwargs):
        if by is not None:
            raise NotImplementedError(
                f"Non-none `by` argument is not yet supported. {constants.FEEDBACK_LINK}"
            )
        if kwargs.pop("backend", None) is not None:
            raise NotImplementedError(
                f"Only support matplotlib backend for now. {constants.FEEDBACK_LINK}"
            )
        return plotbackend.plot(self._parent.copy(), kind="hist", **kwargs)
