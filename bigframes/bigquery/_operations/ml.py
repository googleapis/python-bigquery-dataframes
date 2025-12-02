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

from __future__ import annotations

from typing import Mapping, Optional, TYPE_CHECKING, Union

import bigframes.core.log_adapter as log_adapter
import bigframes.core.sql.ml
import bigframes.dataframe as dataframe

if TYPE_CHECKING:
    import bigframes.ml.base
    import bigframes.session


# Helper to convert DataFrame to SQL string
def _to_sql(df_or_sql: Union[dataframe.DataFrame, str]) -> str:
    if isinstance(df_or_sql, str):
        return df_or_sql
    # It's a DataFrame
    sql, _, _ = df_or_sql._to_sql_query(include_index=False)
    return sql


@log_adapter.method_logger(custom_base_name="bigquery_ml")
def create_model(
    model_name: str,
    *,
    replace: bool = False,
    if_not_exists: bool = False,
    transform: Optional[list[str]] = None,
    input_schema: Optional[Mapping[str, str]] = None,
    output_schema: Optional[Mapping[str, str]] = None,
    connection_name: Optional[str] = None,
    options: Optional[Mapping[str, Union[str, int, float, bool, list]]] = None,
    training_data: Optional[Union[dataframe.DataFrame, str]] = None,
    custom_holiday: Optional[Union[dataframe.DataFrame, str]] = None,
    session: Optional[bigframes.session.Session] = None,
) -> bigframes.ml.base.BaseEstimator:
    """
    Creates a BigQuery ML model.
    """
    import bigframes.pandas as bpd

    training_data_sql = _to_sql(training_data) if training_data is not None else None
    custom_holiday_sql = _to_sql(custom_holiday) if custom_holiday is not None else None

    # Determine session from DataFrames if not provided
    if session is None:
        # Try to get session from inputs
        dfs = [
            obj
            for obj in [training_data, custom_holiday]
            if isinstance(obj, dataframe.DataFrame)
        ]
        if dfs:
            session = dfs[0]._session

    sql = bigframes.core.sql.ml.create_model_ddl(
        model_name=model_name,
        replace=replace,
        if_not_exists=if_not_exists,
        transform=transform,
        input_schema=input_schema,
        output_schema=output_schema,
        connection_name=connection_name,
        options=options,
        training_data=training_data_sql,
        custom_holiday=custom_holiday_sql,
    )

    if session is None:
        session = bpd.get_global_session()

    # Use _start_query_ml_ddl which is designed for this
    session._start_query_ml_ddl(sql)

    return session.read_gbq_model(model_name)
