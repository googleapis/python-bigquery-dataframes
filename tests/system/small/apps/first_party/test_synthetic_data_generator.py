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

import pytest

from bigframes.apps.first_party.synthetic_data_generator.synthetic_data_generator import (
    SyntheticDataGenerator,
)

import bigframes.pandas as bpd

@pytest.mark.parametrize("num_rows", [100, 10001])
def test_generation_from_table(scalars_df_index, num_rows):
    scalars_df_index = scalars_df_index[["float64_col", "int64_col", "int64_too"]]
    df_gen = SyntheticDataGenerator()
    df_gen.generate_synthetic_data_from_table(scalars_df_index, num_rows=num_rows)

    assert len(df_gen.generated_df) == num_rows
    assert len(df_gen.generated_df.columns) == 3


# @pytest.mark.parametrize("num_rows", [3000])
# def test_generation_from_table_1(num_rows):
#     orig_df_1 = bpd.read_gbq("bigquery-public-data.ml_datasets.penguins", max_results=2000)
#     df_gen = SyntheticDataGenerator()
#     df_gen.generate_synthetic_data_from_table(orig_df_1, num_rows=num_rows)

#     assert len(df_gen.generated_df) == num_rows
#     assert len(df_gen.generated_df.columns) == 3


@pytest.mark.parametrize("num_rows", [100, 10001])
def test_generation_with_extra_column_from_table(scalars_df_index, num_rows):
    correlated_column = scalars_df_index["string_col"]
    scalars_df_index = scalars_df_index[["float64_col", "int64_col", "int64_too"]]

    df_gen = SyntheticDataGenerator()
    df_gen.generate_synthetic_correlated_data_from_table(
        scalars_df_index, num_rows=num_rows, correlated_column=correlated_column
    )

    assert len(df_gen.generated_df) == num_rows
    assert len(df_gen.generated_df.columns) == 4


@pytest.mark.parametrize("num_rows", [100, 10001])
def test_generation_from_dict(num_rows):
    dataframe_schema_dict = {
        "num_rows": num_rows,
        "columns": [
            ("name", "str", ""),
            ("age", "int", "less than 90"),
        ],
    }
    df_gen = SyntheticDataGenerator()
    df_gen.generate_synthetic_data(dataframe_schema_dict=dataframe_schema_dict)

    assert len(df_gen.generated_df) == num_rows
    assert len(df_gen.generated_df.columns) == 2


@pytest.mark.parametrize("num_rows", [100, 10001])
def test_generation_with_extra_column_from_dict(scalars_df_index, num_rows):
    dataframe_schema_dict = {
        "num_rows": num_rows,
        "columns": [
            ("name", "str", ""),
            ("age", "int", "less than 90"),
        ],
    }

    correlated_column = scalars_df_index["string_col"]
    df_gen = SyntheticDataGenerator()
    df_gen.generate_synthetic_correlated_data(
        dataframe_schema_dict=dataframe_schema_dict, correlated_column=correlated_column
    )

    assert len(df_gen.generated_df) == num_rows
    assert len(df_gen.generated_df.columns) == 3
