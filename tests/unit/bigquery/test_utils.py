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

import numpy as np
import pandas as pd

import bigframes.bigquery as bbq
import bigframes.pandas as bpd


def test_create_vector_search_sql_simple():
    sql_string = "SELECT embedding FROM my_embeddings_table WHERE id = 1"
    options = {
        "base_table": "my_base_table",
        "column_to_search": "my_embedding_column",
        "distance_type": "COSINE",
        "top_k": 10,
        "use_brute_force": False,
    }

    expected_query = f"""
    SELECT
        query.*,
        base.*,
        distance,
    FROM VECTOR_SEARCH(
        TABLE `my_base_table`,
        "my_embedding_column",
        ({sql_string}),
        distance_type => "COSINE",
        top_k => 10
    )
    """

    result_query = bbq.utils.create_vector_search_sql(
        sql_string, options  # type:ignore
    )
    assert result_query == expected_query


def test_create_vector_search_sql_query_column_to_search():
    sql_string = "SELECT embedding FROM my_embeddings_table WHERE id = 1"
    options = {
        "base_table": "my_base_table",
        "column_to_search": "my_embedding_column",
        "distance_type": "COSINE",
        "top_k": 10,
        "query_column_to_search": "new_embedding_column",
        "use_brute_force": False,
    }

    expected_query = f"""
    SELECT
        query.*,
        base.*,
        distance,
    FROM VECTOR_SEARCH(
        TABLE `my_base_table`,
        "my_embedding_column",
        ({sql_string}),
        "new_embedding_column",
        distance_type => "COSINE",
        top_k => 10
    )
    """

    result_query = bbq.utils.create_vector_search_sql(
        sql_string, options  # type:ignore
    )
    assert result_query == expected_query


def test_apply_sql_df_query():
    query = bpd.DataFrame(
        {
            "query_id": ["dog", "cat"],
            "embedding": [[1.0, 2.0], [3.0, 5.2]],
        }
    )
    options = {
        "base_table": "bigframes-dev.bigframes_tests_sys.base_table",
        "column_to_search": "my_embedding",
        "distance_type": "cosine",
        "top_k": 2,
    }
    result = bbq.utils.apply_sql(query, options).to_pandas()  # type:ignore
    expected = pd.DataFrame(
        {
            "query_id": ["cat", "dog", "dog", "cat"],
            "embedding": [
                np.array([3.0, 5.2]),
                np.array([1.0, 2.0]),
                np.array([1.0, 2.0]),
                np.array([3.0, 5.2]),
            ],
            "id": [1, 2, 1, 2],
            "my_embedding": [
                np.array([1.0, 2.0]),
                np.array([2.0, 4.0]),
                np.array([1.0, 2.0]),
                np.array([2.0, 4.0]),
            ],
            "distance": [0.001777, 0.0, 0.0, 0.001777],
        },
        index=pd.Index([1, 0, 0, 1], dtype="Int64"),
    )
    pd.testing.assert_frame_equal(result, expected, check_dtype=False, rtol=0.1)


def test_apply_sql_series_query():
    query = bpd.Series([[1.0, 2.0], [3.0, 5.2]])
    options = {
        "base_table": "bigframes-dev.bigframes_tests_sys.base_table",
        "column_to_search": "my_embedding",
        "distance_type": "euclidean",
        "top_k": 2,
    }
    result = bbq.utils.apply_sql(query, options).to_pandas()  # type:ignore
    expected = pd.DataFrame(
        {
            "0": [
                np.array([3.0, 5.2]),
                np.array([1.0, 2.0]),
                np.array([3.0, 5.2]),
                np.array([1.0, 2.0]),
            ],
            "id": [2, 4, 5, 1],
            "my_embedding": [
                np.array([2.0, 4.0]),
                np.array([1.0, 3.2]),
                np.array([5.0, 5.4]),
                np.array([1.0, 2.0]),
            ],
            "distance": [1.562049935181331, 1.2000000000000002, 2.009975124224178, 0.0],
        },
        index=pd.Index([1, 0, 1, 0], dtype="Int64"),
    )
    pd.testing.assert_frame_equal(result, expected, check_dtype=False, rtol=0.1)
