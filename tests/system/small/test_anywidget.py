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

import math

import pytest

import bigframes as bf


def test_repr_anywidget_initial_state(
    penguins_df_default_index: bf.dataframe.DataFrame,
):
    pytest.importorskip("anywidget")
    with bf.option_context("display.repr_mode", "anywidget"):
        from bigframes.display import TableWidget

        widget = TableWidget(penguins_df_default_index)
        assert widget.page == 0
        assert widget.page_size == bf.options.display.max_rows
        assert widget.row_count > 0


def test_repr_anywidget_pagination_navigation(
    penguins_df_default_index: bf.dataframe.DataFrame,
):
    """Test basic prev/next navigation functionality."""
    pytest.importorskip("anywidget")
    with bf.option_context("display.repr_mode", "anywidget"):
        from bigframes.display.anywidget import TableWidget

        widget = TableWidget(penguins_df_default_index)

        # Test initial state
        assert widget.page == 0

        # Simulate next page click
        widget.page = 1
        assert widget.page == 1

        # Simulate prev page click
        widget.page = 0
        assert widget.page == 0


def test_repr_anywidget_pagination_edge_cases(
    penguins_df_default_index: bf.dataframe.DataFrame,
):
    """Test pagination at boundaries."""
    pytest.importorskip("anywidget")
    with bf.option_context("display.repr_mode", "anywidget"):
        from bigframes.display.anywidget import TableWidget

        widget = TableWidget(penguins_df_default_index)

        # Test going below page 0
        widget.page = -1
        # Should stay at 0 (handled by frontend)

        # Test going beyond last page
        total_pages = math.ceil(widget.row_count / widget.page_size)
        widget.page = total_pages + 1
        # Should be clamped to last valid page


def test_repr_anywidget_pagination_different_page_sizes(
    penguins_df_default_index: bf.dataframe.DataFrame,
):
    """Test pagination with different page sizes."""
    pytest.importorskip("anywidget")

    # Test with smaller page size
    with bf.option_context("display.repr_mode", "anywidget", "display.max_rows", 5):
        from bigframes.display.anywidget import TableWidget

        widget = TableWidget(penguins_df_default_index)

        assert widget.page_size == 5
        total_pages = math.ceil(widget.row_count / 5)
        assert total_pages > 1  # Should have multiple pages

        # Navigate through several pages
        for page in range(min(3, total_pages)):
            widget.page = page
            assert widget.page == page


def test_repr_anywidget_pagination_buttons_functionality(
    penguins_df_default_index: bf.dataframe.DataFrame,
):
    """Test complete pagination button functionality."""
    pytest.importorskip("anywidget")
    with bf.option_context("display.repr_mode", "anywidget", "display.max_rows", 10):
        from bigframes.display.anywidget import TableWidget

        widget = TableWidget(penguins_df_default_index)

        # Test initial state
        assert widget.page == 0
        assert widget.page_size == 10
        assert widget.row_count > 0

        # Calculate expected pages
        total_pages = math.ceil(widget.row_count / widget.page_size)

        # Test navigation through all pages
        for page_num in range(min(total_pages, 5)):  # Test first 5 pages
            widget.page = page_num
            assert widget.page == page_num
            # Verify table_html is updated
            assert len(widget.table_html) > 0
