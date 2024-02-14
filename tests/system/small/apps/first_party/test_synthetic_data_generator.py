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


@pytest.mark.parametrize(
    "num_rows, use_column_values, use_string_column_values",
    [
        (100, False, False),
        (100, False, True),
        (100, True, False),
        (100, ["float64_col"], False),
        (10001, False, False),
    ],
)
def test_generation_from_table(
    scalars_df_index, num_rows, use_column_values, use_string_column_values
):
    scalars_df_index = scalars_df_index[
        [
            "bool_col",
            "datetime_col",
            "int64_col",
            "float64_col",
            "string_col",
            "timestamp_col",
        ]
    ]
    df_gen = SyntheticDataGenerator()
    df_gen.generate_synthetic_data_from_table(
        scalars_df_index,
        num_rows=num_rows,
        use_column_values=use_column_values,
        use_string_column_values=use_string_column_values,
    )

    assert len(df_gen.generated_df) == num_rows
    assert len(df_gen.generated_df.columns) == 6


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


@pytest.mark.parametrize("num_rows", [100, 10001])
def test_generation_from_dict_with_chunk(scalars_df_index, num_rows):
    dataframe_schema_dict = {
        "num_rows": num_rows,
        "columns": [
            ("name", "str", ""),
            ("age", "int", "less than 90"),
            ("gender", "str", ""),
        ],
    }

    for i in range(50):
        column_name = chr(65 + i // 26) + chr(65 + i % 26)
        column_description = "generate random string length 10."
        dataframe_schema_dict["columns"].append(
            (column_name, "str", column_description)
        )

    df_gen = SyntheticDataGenerator()
    df_gen.generate_synthetic_data(dataframe_schema_dict=dataframe_schema_dict)

    expected_columns = ["name", "age", "gender"] + [
        chr(65 + i // 26) + chr(65 + i % 26) for i in range(50)
    ]

    assert len(df_gen.generated_df) == num_rows
    assert df_gen.generated_df.columns.tolist() == expected_columns


@pytest.mark.parametrize("num_rows", [100, 10001])
def test_generation_with_extra_column_from_dict_with_chunk(scalars_df_index, num_rows):
    dataframe_schema_dict = {
        "num_rows": num_rows,
        "columns": [
            ("name", "str", ""),
            ("age", "int", "less than 90"),
            ("gender", "str", ""),
        ],
    }

    correlated_column = scalars_df_index["string_col"]

    for i in range(50):
        column_name = chr(65 + i // 26) + chr(65 + i % 26)
        column_description = "generate random string length 10."
        dataframe_schema_dict["columns"].append(
            (column_name, "str", column_description)
        )

    df_gen = SyntheticDataGenerator()
    df_gen.generate_synthetic_correlated_data(
        dataframe_schema_dict=dataframe_schema_dict, correlated_column=correlated_column
    )

    expected_columns = (
        ["name", "age", "gender"]
        + [chr(65 + i // 26) + chr(65 + i % 26) for i in range(50)]
        + ["string_col"]
    )

    assert len(df_gen.generated_df) == num_rows
    assert df_gen.generated_df.columns.tolist() == expected_columns
