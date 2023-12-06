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

import bigframes
import bigframes.pandas as bpd


def summarize_df_to_dict(df, num_rows, use_string_column_values=False):
    """
    Summarizes a DataFrame into a dictionary format.

    Args:
        df (DataFrame):
            The DataFrame to summarize.
        num_rows (int):
            The number of rows to be used in the synthetic data generation.
        use_string_column_values (bool, optional):
            If True, includes sample string values from each column in the description.
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
    sampled_df = None
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
        elif str(df[col].dtype).startswith("timestamp"):
            col_min = df[col].min()
            col_max = df[col].max()
            column_desc = f"random in range: {col_min} to {col_max}, convert time provided to datetime type first"
            if "UTC" in str(df[col].dtype):
                column_desc += ", use faker with tzinfo=pytz.utc"
        elif use_string_column_values:
            if sampled_df is None:
                sampled_df = df.sample(10).to_pandas()
            uniq_vals = sampled_df[col].unique()
            column_desc = (
                "has values like: " + ", ".join(uniq_vals) if uniq_vals else "None"
            )
        else:
            column_desc = "None"

        columns.append((column_name, column_dtype, column_desc))

    description = {"num_rows": num_rows, "columns": columns}

    return description


def _split_non_numeric(df, keep_bool=True):
    types_to_keep = set(bigframes.dtypes.NUMERIC_BIGFRAMES_TYPES)
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
