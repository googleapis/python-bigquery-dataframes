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

import pandas as pd
import pytest

import bigframes as bf
from bigframes.display import TableWidget

pytest.importorskip("anywidget")


@pytest.fixture(scope="module")
def paginated_pandas_df() -> pd.DataFrame:
    """Create a minimal test DataFrame with exactly 3 pages of 2 rows each."""
    test_data = pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5],
            "page_indicator": [
                # Page 1 (rows 1-2)
                "page_1_row_1",
                "page_1_row_2",
                # Page 2 (rows 3-4)
                "page_2_row_1",
                "page_2_row_2",
                # Page 3 (rows 5-6)
                "page_3_row_1",
                "page_3_row_2",
            ],
            "value": [0, 1, 2, 3, 4, 5],
        }
    )
    return test_data


@pytest.fixture(scope="module")
def paginated_bf_df(
    session: bf.Session, paginated_pandas_df: pd.DataFrame
) -> bf.dataframe.DataFrame:
    return session.read_pandas(paginated_pandas_df)


@pytest.fixture(scope="module")
def table_widget(paginated_bf_df: bf.dataframe.DataFrame) -> TableWidget:
    """
    Helper fixture to create a TableWidget instance with a fixed page size.
    This reduces duplication across tests that use the same widget configuration.
    """
    with bf.option_context("display.repr_mode", "anywidget", "display.max_rows", 2):
        widget = TableWidget(paginated_bf_df)
    return widget


def _assert_html_matches_pandas_slice(
    table_html: str,
    expected_pd_slice: pd.DataFrame,
    full_pd_df: pd.DataFrame,
):
    """
    Assertion helper to verify that the rendered HTML contains exactly the
    rows from the expected pandas DataFrame slice and no others. This is
    inspired by the pattern of comparing BigFrames output to pandas output.
    """
    # Check that the unique indicator from each expected row is present.
    for _, row in expected_pd_slice.iterrows():
        assert row["page_indicator"] in table_html

    # Create a DataFrame of all rows that should NOT be present.
    unexpected_pd_df = full_pd_df.drop(expected_pd_slice.index)

    # Check that no unique indicators from unexpected rows are present.
    for _, row in unexpected_pd_df.iterrows():
        assert row["page_indicator"] not in table_html


def test_repr_anywidget_initialization_sets_page_to_zero(
    paginated_bf_df: bf.dataframe.DataFrame,
):
    """A TableWidget should initialize with the page number set to 0."""
    with bf.option_context("display.repr_mode", "anywidget"):
        from bigframes.display import TableWidget

        widget = TableWidget(paginated_bf_df)

        assert widget.page == 0


def test_repr_anywidget_initialization_sets_page_size_from_options(
    paginated_bf_df: bf.dataframe.DataFrame,
):
    """A TableWidget should initialize its page size from bf.options."""
    with bf.option_context("display.repr_mode", "anywidget"):
        from bigframes.display import TableWidget

        widget = TableWidget(paginated_bf_df)

        assert widget.page_size == bf.options.display.max_rows


def test_repr_anywidget_initialization_sets_row_count(
    paginated_bf_df: bf.dataframe.DataFrame,
    paginated_pandas_df: pd.DataFrame,
):
    """A TableWidget should initialize with the correct total row count."""
    with bf.option_context("display.repr_mode", "anywidget"):
        from bigframes.display import TableWidget

        widget = TableWidget(paginated_bf_df)

        assert widget.row_count == len(paginated_pandas_df)


def test_repr_anywidget_display_first_page_on_load(
    table_widget: TableWidget, paginated_pandas_df: pd.DataFrame
):
    """
    Given a widget, when it is first loaded, then it should display
    the first page of data.
    """
    expected_slice = paginated_pandas_df.iloc[0:2]

    html = table_widget.table_html

    _assert_html_matches_pandas_slice(html, expected_slice, paginated_pandas_df)


def test_repr_anywidget_navigate_to_second_page(
    table_widget: TableWidget, paginated_pandas_df: pd.DataFrame
):
    """
    Given a widget, when the page is set to 1, then it should display
    the second page of data.
    """
    expected_slice = paginated_pandas_df.iloc[2:4]

    table_widget.page = 1
    html = table_widget.table_html

    assert table_widget.page == 1
    _assert_html_matches_pandas_slice(html, expected_slice, paginated_pandas_df)


def test_repr_anywidget_navigate_to_last_page(
    table_widget: TableWidget, paginated_pandas_df: pd.DataFrame
):
    """
    Given a widget, when the page is set to the last page (2),
    then it should display the final page of data.
    """
    expected_slice = paginated_pandas_df.iloc[4:6]

    table_widget.page = 2
    html = table_widget.table_html

    assert table_widget.page == 2
    _assert_html_matches_pandas_slice(html, expected_slice, paginated_pandas_df)


def test_repr_anywidget_page_clamp_to_zero_for_negative_input(
    table_widget: TableWidget, paginated_pandas_df: pd.DataFrame
):
    """
    Given a widget, when a negative page number is set,
    then the page number should be clamped to 0 and display the first page.
    """
    expected_slice = paginated_pandas_df.iloc[0:2]

    table_widget.page = -1
    html = table_widget.table_html

    assert table_widget.page == 0
    _assert_html_matches_pandas_slice(html, expected_slice, paginated_pandas_df)


def test_repr_anywidget_page_clamp_to_last_page_for_out_of_bounds_input(
    table_widget: TableWidget, paginated_pandas_df: pd.DataFrame
):
    """
    Given a widget, when a page number greater than the max is set,
    then the page number should be clamped to the last valid page.
    """
    expected_slice = paginated_pandas_df.iloc[4:6]

    table_widget.page = 100
    html = table_widget.table_html

    assert table_widget.page == 2
    _assert_html_matches_pandas_slice(html, expected_slice, paginated_pandas_df)


@pytest.mark.parametrize(
    "page, start_row, end_row",
    [
        (0, 0, 3),  # Page 0: rows 0-2
        (1, 3, 6),  # Page 1: rows 3-5
    ],
    ids=[
        "Page 0 (Rows 0-2)",
        "Page 1 (Rows 3-5)",
    ],
)
def test_repr_anywidget_paginate_correctly_with_custom_page_size(
    paginated_bf_df: bf.dataframe.DataFrame,
    paginated_pandas_df: pd.DataFrame,
    page: int,
    start_row: int,
    end_row: int,
):
    """
    A widget should paginate correctly with a custom page size of 3.
    """
    with bf.option_context("display.repr_mode", "anywidget", "display.max_rows", 3):
        from bigframes.display import TableWidget

        widget = TableWidget(paginated_bf_df)
        assert widget.page_size == 3

        expected_slice = paginated_pandas_df.iloc[start_row:end_row]

        widget.page = page
        html = widget.table_html

        assert widget.page == page
        _assert_html_matches_pandas_slice(html, expected_slice, paginated_pandas_df)
