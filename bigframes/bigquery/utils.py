# Copyright 2024 Google LLC
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

from typing import Iterable, Mapping, Union

import bigframes.ml.utils as utils
import bigframes.pandas as bpd


def create_vector_search_sql(
    sql_string: str,
    options: Mapping[str, Union[str, int, float, Iterable[str]]] = {},
) -> str:
    """Encode the VECTOR SEARCH statement for BigQuery Vector Search."""

    base_table = options["base_table"]
    column_to_search = options["column_to_search"]
    distance_type = options["distance_type"]
    top_k = options["top_k"]
    query_column_to_search = options.get("query_column_to_search", None)

    if query_column_to_search is not None:
        query_str = f"""
    SELECT
        query.*,
        base.*,
        distance,
    FROM VECTOR_SEARCH(
        TABLE `{base_table}`,
        "{column_to_search}",
        ({sql_string}),
        "{query_column_to_search}",
        distance_type => "{distance_type}",
        top_k => {top_k}
    )
    """
    else:
        query_str = f"""
    SELECT
        query.*,
        base.*,
        distance,
    FROM VECTOR_SEARCH(
        TABLE `{base_table}`,
        "{column_to_search}",
        ({sql_string}),
        distance_type => "{distance_type}",
        top_k => {top_k}
    )
    """
    return query_str


def apply_sql(
    query: Union[bpd.DataFrame, bpd.Series],
    options: Mapping[str, Union[str, int, float, Iterable[str]]] = {},
) -> bpd.DataFrame:
    """Helper to wrap a dataframe in a SQL query, keeping the index intact.

    Args:
        query (bigframes.dataframe.DataFrame):
            The dataframe to be wrapped.
    """
    (query,) = utils.convert_to_dataframe(query)
    sql_string, index_col_ids, index_labels = query._to_sql_query(include_index=True)

    sql = create_vector_search_sql(sql_string=sql_string, options=options)
    if index_col_ids is not None:
        df = query._session.read_gbq(sql, index_col=index_col_ids)
    else:
        df = query._session.read_gbq(sql)
    df.index.names = index_labels

    return df
