# Contains code from https://github.com/pandas-dev/pandas/blob/main/pandas/io/gbq.py
""" Google BigQuery support """

from __future__ import annotations

from typing import Iterable, Optional

from bigframes import constants


class GBQIOMixin:
    def read_gbq(
        self,
        query_or_table: str,
        *,
        index_col: Iterable[str] | str = (),
        col_order: Iterable[str] = (),
        max_results: Optional[int] = None,
    ):
        """Loads a DataFrame from BigQuery.

        BigQuery tables are an unordered, unindexed data source. By default,
        the DataFrame will have an arbitrary index and ordering.

        Set the `index_col` argument to one or more columns to choose an
        index. The resulting DataFrame is sorted by the index columns. For the
        best performance, ensure the index columns don't contain duplicate
        values.

        .. note::
            By default, even SQL query inputs with an ORDER BY clause create a
            DataFrame with an arbitrary ordering. Use ``row_number() OVER
            (ORDER BY ...) AS rowindex`` in your SQL query and set
            ``index_col='rowindex'`` to preserve the desired ordering.

            If your query doesn't have an ordering, select ``GENERATE_UUID() AS
            rowindex`` in your SQL and set ``index_col='rowindex'`` for the
            best performance.

        **Examples:**

            >>> import bigframes.pandas as bpd
            >>> bpd.options.display.progress_bar = None

        If the input is a table ID:

            >>> bpd.read_gbq("bigquery-public-data.ml_datasets.penguins").head(5)
                                                 species island  culmen_length_mm  \\
            0        Adelie Penguin (Pygoscelis adeliae)  Dream              36.6
            1        Adelie Penguin (Pygoscelis adeliae)  Dream              39.8
            2        Adelie Penguin (Pygoscelis adeliae)  Dream              40.9
            3  Chinstrap penguin (Pygoscelis antarctica)  Dream              46.5
            4        Adelie Penguin (Pygoscelis adeliae)  Dream              37.3
            <BLANKLINE>
               culmen_depth_mm  flipper_length_mm  body_mass_g     sex
            0             18.4              184.0       3475.0  FEMALE
            1             19.1              184.0       4650.0    MALE
            2             18.9              184.0       3900.0    MALE
            3             17.9              192.0       3500.0  FEMALE
            4             16.8              192.0       3000.0  FEMALE
            <BLANKLINE>
            [5 rows x 7 columns]

        Preserve ordering in a query input.

            >>> bpd.read_gbq('''
            ...    SELECT
            ...       -- Instead of an ORDER BY clause on the query, use
            ...       -- ROW_NUMBER() to create an ordered DataFrame.
            ...       ROW_NUMBER() OVER (ORDER BY AVG(pitchSpeed) DESC)
            ...         AS rowindex,
            ...
            ...       pitcherFirstName,
            ...       pitcherLastName,
            ...       AVG(pitchSpeed) AS averagePitchSpeed
            ...     FROM `bigquery-public-data.baseball.games_wide`
            ...     WHERE year = 2016
            ...     GROUP BY pitcherFirstName, pitcherLastName
            ... ''', index_col="rowindex").head(n=5)
                     pitcherFirstName pitcherLastName  averagePitchSpeed
            rowindex
            1                Albertin         Chapman          96.514113
            2                 Zachary         Britton          94.591039
            3                  Trevor       Rosenthal          94.213953
            4                    Jose          Torres          94.103448
            5                  Tayron        Guerrero          93.863636
            <BLANKLINE>
            [5 rows x 3 columns]

        Args:
            query_or_table (str):
                A SQL string to be executed or a BigQuery table to be read. The
                table must be specified in the format of
                `project.dataset.tablename` or `dataset.tablename`.
            index_col (Iterable[str] or str):
                Name of result column(s) to use for index in results DataFrame.
            col_order (Iterable[str]):
                List of BigQuery column names in the desired order for results
                DataFrame.
            max_results (Optional[int], default None):
                If set, limit the maximum number of rows to fetch from the
                query results.

        Returns:
            bigframes.dataframe.DataFrame: A DataFrame representing results of the query or table.
        """
        raise NotImplementedError(constants.ABSTRACT_METHOD_ERROR_MESSAGE)
