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

pytest.importorskip("anywidget")

# Test constants to avoid change detector tests
EXPECTED_ROW_COUNT = 6
EXPECTED_PAGE_SIZE = 2
EXPECTED_TOTAL_PAGES = 3


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


@pytest.fixture
def table_widget(paginated_bf_df: bf.dataframe.DataFrame):
    """
    Helper fixture to create a TableWidget instance with a fixed page size.
    This reduces duplication across tests that use the same widget configuration.
    """
    from bigframes import display

    with bf.option_context("display.repr_mode", "anywidget", "display.max_rows", 2):
        # Delay context manager cleanup of `max_rows` until after tests finish.
        yield display.TableWidget(paginated_bf_df)


@pytest.fixture(scope="module")
def small_pandas_df() -> pd.DataFrame:
    """Create a DataFrame smaller than the page size for edge case testing."""
    return pd.DataFrame(
        {
            "id": [0, 1],
            "page_indicator": ["small_row_1", "small_row_2"],
            "value": [0, 1],
        }
    )


@pytest.fixture(scope="module")
def small_bf_df(
    session: bf.Session, small_pandas_df: pd.DataFrame
) -> bf.dataframe.DataFrame:
    return session.read_pandas(small_pandas_df)


@pytest.fixture
def small_widget(small_bf_df):
    """Helper fixture for tests using a DataFrame smaller than the page size."""
    from bigframes import display

    with bf.option_context("display.repr_mode", "anywidget", "display.max_rows", 5):
        yield display.TableWidget(small_bf_df)


@pytest.fixture(scope="module")
def empty_pandas_df() -> pd.DataFrame:
    """Create an empty DataFrame for edge case testing."""
    return pd.DataFrame(columns=["id", "page_indicator", "value"])


@pytest.fixture(scope="module")
def empty_bf_df(
    session: bf.Session, empty_pandas_df: pd.DataFrame
) -> bf.dataframe.DataFrame:
    return session.read_pandas(empty_pandas_df)


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


def test_widget_initialization_should_calculate_total_row_count(
    paginated_bf_df: bf.dataframe.DataFrame,
):
    """A TableWidget should correctly calculate the total row count on creation."""
    from bigframes import display

    with bf.option_context("display.repr_mode", "anywidget", "display.max_rows", 2):
        widget = display.TableWidget(paginated_bf_df)

    assert widget.row_count == EXPECTED_ROW_COUNT


def test_widget_initialization_should_set_default_pagination(
    table_widget,
):
    """A TableWidget should initialize with page 0 and the correct page size."""
    # The `table_widget` fixture already creates the widget.
    # Assert its state.
    assert table_widget.page == 0
    assert table_widget.page_size == EXPECTED_PAGE_SIZE


def test_widget_display_should_show_first_page_on_load(
    table_widget, paginated_pandas_df: pd.DataFrame
):
    """
    Given a widget, when it is first loaded, then it should display
    the first page of data.
    """
    expected_slice = paginated_pandas_df.iloc[0:2]

    html = table_widget.table_html

    _assert_html_matches_pandas_slice(html, expected_slice, paginated_pandas_df)


@pytest.mark.parametrize(
    "page_number, start_row, end_row",
    [
        (1, 2, 4),  # Second page
        (2, 4, 6),  # Last page
    ],
    ids=["second_page", "last_page"],
)
def test_widget_navigation_should_display_correct_page(
    table_widget,
    paginated_pandas_df: pd.DataFrame,
    page_number: int,
    start_row: int,
    end_row: int,
):
    """
    Given a widget, when the page is set, then it should display the correct
    slice of data.
    """
    expected_slice = paginated_pandas_df.iloc[start_row:end_row]

    table_widget.page = page_number
    html = table_widget.table_html

    assert table_widget.page == page_number
    _assert_html_matches_pandas_slice(html, expected_slice, paginated_pandas_df)


def test_widget_navigation_should_clamp_to_zero_for_negative_input(
    table_widget, paginated_pandas_df: pd.DataFrame
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


def test_widget_navigation_should_clamp_to_last_page_for_out_of_bounds_input(
    table_widget, paginated_pandas_df: pd.DataFrame
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
def test_widget_pagination_should_work_with_custom_page_size(
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


def test_widget_with_few_rows_should_display_all_rows(small_widget, small_pandas_df):
    """
    Given a DataFrame smaller than the page size, the widget should
    display all rows on the first page.
    """
    html = small_widget.table_html

    _assert_html_matches_pandas_slice(html, small_pandas_df, small_pandas_df)


def test_widget_with_few_rows_should_have_only_one_page(small_widget):
    """
    Given a DataFrame smaller than the page size, the widget should
    clamp page navigation, effectively having only one page.
    """
    assert small_widget.page == 0

    # Attempt to navigate past the end
    small_widget.page = 1

    # Should be clamped back to the only valid page
    assert small_widget.page == 0


def test_widget_page_size_should_be_immutable_after_creation(
    paginated_bf_df: bf.dataframe.DataFrame,
):
    """
    A widget's page size should be fixed on creation and not be affected
    by subsequent changes to global options.
    """
    with bf.option_context("display.repr_mode", "anywidget", "display.max_rows", 2):
        from bigframes.display import TableWidget

        widget = TableWidget(paginated_bf_df)
        assert widget.page_size == 2

        # Navigate to second page to ensure widget is in a non-default state
        widget.page = 1
        assert widget.page == 1

        # Change global max_rows - widget should not be affected
        bf.options.display.max_rows = 10

        assert widget.page_size == 2  # Should remain unchanged
        assert widget.page == 1  # Should remain on same page


def test_empty_widget_should_have_zero_row_count(empty_bf_df: bf.dataframe.DataFrame):
    """Given an empty DataFrame, the widget's row count should be 0."""
    with bf.option_context("display.repr_mode", "anywidget"):
        from bigframes.display import TableWidget

        widget = TableWidget(empty_bf_df)

        assert widget.row_count == 0


def test_empty_widget_should_render_table_headers(empty_bf_df: bf.dataframe.DataFrame):
    """Given an empty DataFrame, the widget should still render table headers."""
    with bf.option_context("display.repr_mode", "anywidget"):
        from bigframes.display import TableWidget

        widget = TableWidget(empty_bf_df)

        html = widget.table_html

        assert "<table" in html
        assert "id" in html


def test_page_size_change_should_reset_current_page_to_zero(table_widget):
    """
    Given a widget on a non-default page, When the page_size is changed,
    Then the current page attribute should reset to 0.
    """
    # Start on page 1 with an initial page size of 2.
    table_widget.page = 1
    assert table_widget.page == 1

    # Change the page size.
    table_widget.page_size = 3

    # The page number is reset to 0.
    assert table_widget.page == 0


def test_page_size_change_should_render_html_with_new_size(
    table_widget, paginated_pandas_df: pd.DataFrame
):
    """
    Given a widget, when the page_size is changed,
    the rendered HTML should immediately reflect the new page size.
    """
    # The widget is in its initial state with page_size=2.
    # We expect the first 3 rows after the change.
    expected_slice = paginated_pandas_df.iloc[0:3]

    # Change the page size.
    table_widget.page_size = 3

    # The HTML now contains the first 3 rows.
    html = table_widget.table_html
    _assert_html_matches_pandas_slice(html, expected_slice, paginated_pandas_df)


def test_navigation_after_page_size_change_should_use_new_size(
    table_widget, paginated_pandas_df: pd.DataFrame
):
    """
    Given a widget whose page size has been changed, When we navigate to the
    next page, Then the pagination should use the new page size.
    """
    # Change the page size to 3.
    table_widget.page_size = 3
    # We expect the second page to contain rows 4-6 (indices 3-6).
    expected_slice = paginated_pandas_df.iloc[3:6]

    # Navigate to the next page.
    table_widget.page = 1

    # The second page's HTML correctly reflects the new page size.
    html = table_widget.table_html
    _assert_html_matches_pandas_slice(html, expected_slice, paginated_pandas_df)


@pytest.mark.parametrize("invalid_size", [0, -5], ids=["zero", "negative"])
def test_setting_invalid_page_size_should_be_ignored(table_widget, invalid_size: int):
    """When the page size is set to an invalid number (<=0), the change should
    be ignored."""
    # Set the initial page to 2.
    initial_size = table_widget.page_size
    assert initial_size == 2

    # Attempt to set the page size to a invlaid size.
    table_widget.page_size = invalid_size

    # The page size remains unchanged.
    assert table_widget.page_size == initial_size


def test_setting_page_size_above_max_should_be_clamped(table_widget):
    """
    Given a widget, when the page size is set to a value greater than the
    allowed maximum, the page size should be clamped to the maximum value.
    """
    # The maximum is hardcoded to 1000 in the implementation.
    expected_clamped_size = 1000

    # Attempt to set a very large page size.
    table_widget.page_size = 9001

    # The page size is clamped to the maximum.
    assert table_widget.page_size == expected_clamped_size


def test_widget_creation_should_load_css_for_rendering(table_widget):
    """
    Given a TableWidget is created, when its resources are accessed,
    it should contain the CSS content required for styling.
    """
    # The table_widget fixture creates the widget.
    # No additional setup is needed.

    # Access the CSS content.
    css_content = table_widget._css

    # The content is a non-empty string containing a known selector.
    assert isinstance(css_content, str)
    assert len(css_content) > 0
    assert ".bigframes-widget .footer" in css_content


# TODO(shuowei): Add tests for custom index and multiindex
# This may not be necessary for the SQL Cell use case but should be
# considered for completeness.
