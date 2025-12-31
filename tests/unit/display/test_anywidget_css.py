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

import unittest.mock as mock

import bigframes.dataframe
from bigframes.display.anywidget import TableWidget


def test_css_styles_traitlet_is_populated():
    """Verify that css_styles traitlet is populated from the external CSS file."""
    # Mock the dataframe and its block
    mock_df = mock.create_autospec(bigframes.dataframe.DataFrame, instance=True)
    mock_df.columns = ["col1"]
    mock_df.dtypes = {"col1": "object"}
    mock_block = mock.Mock()
    mock_block.has_index = False
    mock_df._block = mock_block

    # Mock _initial_load to avoid side effects
    with mock.patch.object(TableWidget, "_initial_load"):
        widget = TableWidget(mock_df)

        # Check that css_styles is not empty
        assert widget.css_styles

        # Check that it contains expected CSS content (e.g. dark mode query)
        assert "@media (prefers-color-scheme: dark)" in widget.css_styles
        assert ".bigframes-widget" in widget.css_styles
