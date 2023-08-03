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

import typing
from typing import Iterable, Union

import bigframes.constants as constants
from bigframes.core import blocks
import bigframes.pandas as bpd

# Internal type alias
FrameType = Union[bpd.DataFrame, bpd.Series]


def convert_to_dataframe(*input: FrameType) -> Iterable[bpd.DataFrame]:
    return (_convert_to_dataframe(frame) for frame in input)


def _convert_to_dataframe(frame: FrameType) -> bpd.DataFrame:
    if isinstance(frame, bpd.DataFrame):
        return frame
    if isinstance(frame, bpd.Series):
        return frame.to_frame()
    raise ValueError(
        f"Unsupported type {type(frame)} to convert to DataFrame. {constants.FEEDBACK_LINK}"
    )


def convert_to_series(*input: FrameType) -> Iterable[bpd.Series]:
    return (_convert_to_series(frame) for frame in input)


def _convert_to_series(frame: FrameType) -> bpd.Series:
    if isinstance(frame, bpd.DataFrame):
        if len(frame.columns) != 1:
            raise ValueError(
                "To convert into Series, DataFrames can only contain one column. "
                f"Try input with only one column. {constants.FEEDBACK_LINK}"
            )

        label = typing.cast(blocks.Label, frame.columns.tolist()[0])
        return typing.cast(bpd.Series, frame[label])
    if isinstance(frame, bpd.Series):
        return frame
    raise ValueError(
        f"Unsupported type {type(frame)} to convert to Series. {constants.FEEDBACK_LINK}"
    )
