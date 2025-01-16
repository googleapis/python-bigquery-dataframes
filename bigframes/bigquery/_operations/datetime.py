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

from bigframes import operations as ops
from bigframes import series


def unix_seconds(input: series.Series) -> series.Series:
    return input._apply_unary_op(ops.UnixSeconds())


def unix_millis(input: series.Series) -> series.Series:
    return input._apply_unary_op(ops.UnixMillis())


def unix_micros(input: series.Series) -> series.Series:
    return input._apply_unary_op(ops.UnixMicros())
