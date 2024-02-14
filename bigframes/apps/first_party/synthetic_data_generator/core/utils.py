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

from datetime import timezone
from typing import List

import pandas as pd

import bigframes
import bigframes.pandas as bpd


def summarize_df_to_dict(df, num_rows, use_column_values=False):
    """
    Summarizes a DataFrame into a dictionary format.

    Args:
        df (DataFrame):
            The DataFrame to summarize.
        num_rows (int):
            The number of rows to be used in the synthetic data generation.
        use_string_column_values (bool, str, list[str], optional):
            If True, includes sample values in the description.
            Defaults to False.

    Returns:
        Dict[str, Any]:
            A dictionary containing the number of rows and a detailed description of each column.
            The description includes column names, data types, and additional details like
            statistical summaries or sample values.
    """
    df_numeric, _ = _split_non_numeric(df, keep_bool=False)
    stats = None
    if len(df_numeric.columns) != 0:
        stats = df_numeric.describe().to_pandas()

    columns = []
    sampled_df = df.sample(10).dropna().to_pandas() if use_column_values else None

    if isinstance(use_column_values, List) and not all(
        column in sampled_df.columns for column in use_column_values
    ):
        missing_cols = [
            column for column in use_column_values if column not in sampled_df.columns
        ]
        raise ValueError(
            f"Columns {missing_cols} specified in 'use_column_values' do not exist in the DataFrame. "
            "Check the 'use_column_values' list for accuracy."
        )

    dtype_mapping = {
        "timestamp[us, tz=UTC][pyarrow]": "pd.Timestamp UTC",
        "timestamp[us][pyarrow]": "pd.Timestamp",
    }

    for col in df.columns:
        column_name = col
        column_dtype = (
            str(df[col].dtype)
            if str(df[col].dtype) not in dtype_mapping
            else dtype_mapping[str(df[col].dtype)]
        )

        if stats is not None and col in stats.columns:
            column_desc_items = [
                (stat, value) for stat, value in stats[col].items() if stat != "count"
            ]
            column_desc = ", ".join(
                [f"{stat}: {value:.2f}" for stat, value in column_desc_items]
            )
            column_desc += _get_column_sample(
                sampled_df, col, use_column_values, has_precedent=True
            )
        elif str(df[col].dtype).startswith("timestamp"):
            col_min = df[col].min()
            col_max = df[col].max()
            column_desc = f"random in range: {col_min} to {col_max}, convert time provided to datetime type first"
            if "UTC" in str(df[col].dtype):
                column_desc += ", this column should be with UTC timezone"
            else:
                column_desc += ", this column should be with no timezone"
            column_desc += _get_column_sample(
                sampled_df, col, use_column_values, has_precedent=True
            )
        else:
            column_desc = _get_column_sample(sampled_df, col, use_column_values)

        columns.append((column_name, column_dtype, column_desc))

    description = {"num_rows": num_rows, "columns": columns}

    return description


def _get_column_sample(
    sampled_df: bpd.DataFrame, col: str, use_column_values, has_precedent=False
):
    if (
        use_column_values == False
        or (use_column_values == "string" and sampled_df[col].dtype != "string")
        or (isinstance(use_column_values, List) and col not in use_column_values)
    ):
        return "" if has_precedent else "None"

    if (
        not isinstance(use_column_values, (bool, List))
        and use_column_values != "string"
    ):
        raise ValueError(
            f"Invalid 'use_column_values' argument value: {use_column_values}. "
            "Expected 'True', 'False', or a list of valid column names."
        )

    uniq_vals = sampled_df[col].unique()
    column_desc = (
        (", " if has_precedent else "")
        + "has values like: "
        + ", ".join([repr(uniq_val) for uniq_val in uniq_vals])
        + ", use random_element."
        if uniq_vals
        else "None"
    )
    return column_desc


def _split_non_numeric(df, keep_bool=True):
    types_to_keep = set(bigframes.dtypes.NUMERIC_BIGFRAMES_TYPES_RESTRICTIVE)
    if not keep_bool:
        types_to_keep -= set(bigframes.dtypes.BOOL_BIGFRAMES_TYPES)
    non_numeric_cols = [
        col_id
        for col_id, dtype in zip(df._block.value_columns, df._block.dtypes)
        if dtype not in types_to_keep
    ]
    return bpd.DataFrame(df._block.drop_columns(non_numeric_cols)), bpd.DataFrame(
        df._block.select_columns(non_numeric_cols)
    )


def is_notebook() -> bool:
    """
    Check if the Python code is running in a Jupyter notebook environment.

    Returns:
        bool: True if running in a Jupyter notebook, False otherwise.
    """
    try:
        from IPython import get_ipython

        if "IPKernelApp" not in get_ipython().config:  # pragma: no cover
            return False  # Not in a notebook
    except (ImportError, AttributeError):
        return False  # IPython not available or not in a notebook

    return True
