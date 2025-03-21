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

from bigframes.core import window_spec


def test_window_boundary_preceding():
    window = window_spec.WindowBoundary.preceding(1)

    assert window == window_spec.WindowBoundary(1, is_preceding=True)


def test_window_boundary_following():
    window = window_spec.WindowBoundary.following(1)

    assert window == window_spec.WindowBoundary(1, is_preceding=False)
