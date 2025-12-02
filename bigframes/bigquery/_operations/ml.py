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

from typing import Mapping, Optional, Union

import bigframes.core.log_adapter as log_adapter
import bigframes.core.sql.ml
import bigframes.dataframe as dataframe
import bigframes.ml.base
import bigframes.session


# Helper to convert DataFrame to SQL string
def _to_sql(df_or_sql: Union[dataframe.DataFrame, str]) -> str:
    if isinstance(df_or_sql, str):
        return df_or_sql
    # It's a DataFrame
    sql, _, _ = df_or_sql._to_sql_query(include_index=False)
    return sql


def _get_model_name_and_session(
    model: Union[bigframes.ml.base.BaseEstimator, str],
    # Other dataframe arguments to extract session from
    *dataframes: Optional[Union[dataframe.DataFrame, str]],
) -> tuple[str, bigframes.session.Session]:
    import bigframes.pandas as bpd

    if isinstance(model, str):
        model_name = model
        session = None
        for df in dataframes:
            if isinstance(df, dataframe.DataFrame):
                session = df._session
                break
        if session is None:
            session = bpd.get_global_session()
        return model_name, session
    else:
        if model._bqml_model is None:
            raise ValueError("Model must be fitted to be used in ML operations.")
        return model._bqml_model.model_name, model._bqml_model.session


@log_adapter.method_logger(custom_base_name="bigquery_ml")
def create_model(
    model_name: str,
    *,
    replace: bool = False,
    if_not_exists: bool = False,
    # TODO(tswast): Also support bigframes.ml transformer classes and/or
    # bigframes.pandas functions?
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

    See the `BigQuery ML CREATE MODEL DDL syntax
    <https://docs.cloud.google.com/bigquery/docs/reference/standard-sql/bigqueryml-syntax-create>`_
    for additional reference.

    Args:
        model_name (str):
            The name of the model in BigQuery.
        replace (bool, default False):
            Whether to replace the model if it already exists.
        if_not_exists (bool, default False):
            Whether to ignore the error if the model already exists.
        transform (list[str], optional):
            A list of SQL transformations for the TRANSFORM clause, which
            specifies the preprocessing steps to apply to the input data.
        input_schema (Mapping[str, str], optional):
            The INPUT clause, which specifies the schema of the input data.
        output_schema (Mapping[str, str], optional):
            The OUTPUT clause, which specifies the schema of the output data.
        connection_name (str, optional):
            The connection to use for the model.
        options (Mapping[str, Union[str, int, float, bool, list]], optional):
            The OPTIONS clause, which specifies the model options.
        training_data (Union[bigframes.pandas.DataFrame, str], optional):
            The query or DataFrame to use for training the model.
        custom_holiday (Union[bigframes.pandas.DataFrame, str], optional):
            The query or DataFrame to use for custom holiday data.
        session (bigframes.session.Session, optional):
            The session to use. If not provided, the default session is used.

    Returns:
        bigframes.ml.base.BaseEstimator:
            The created BigQuery ML model.
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


@log_adapter.method_logger(custom_base_name="bigquery_ml")
def evaluate(
    model: Union[bigframes.ml.base.BaseEstimator, str],
    input_: Optional[Union[dataframe.DataFrame, str]] = None,
    *,
    options: Optional[Mapping[str, Union[str, int, float, bool, list]]] = None,
) -> dataframe.DataFrame:
    """
    Evaluates a BigQuery ML model.

    See the `BigQuery ML EVALUATE function syntax
    <https://docs.cloud.google.com/bigquery/docs/reference/standard-sql/bigqueryml-syntax-evaluate>`_
    for additional reference.

    Args:
        model (bigframes.ml.base.BaseEstimator or str):
            The model to evaluate.
        input_ (Union[bigframes.pandas.DataFrame, str], optional):
            The DataFrame or query to use for evaluation. If not provided, the
            evaluation data from training is used.
        options (Mapping[str, Union[str, int, float, bool, list]], optional):
            The OPTIONS clause, which specifies the model options.

    Returns:
        bigframes.pandas.DataFrame:
            The evaluation results.
    """
    model_name, session = _get_model_name_and_session(model, input_)
    table_sql = _to_sql(input_) if input_ is not None else None

    sql = bigframes.core.sql.ml.evaluate(
        model_name=model_name,
        table=table_sql,
        options=options,
    )

    return session.read_gbq(sql)


@log_adapter.method_logger(custom_base_name="bigquery_ml")
def predict(
    model: Union[bigframes.ml.base.BaseEstimator, str],
    input_: Union[dataframe.DataFrame, str],
    *,
    options: Optional[Mapping[str, Union[str, int, float, bool, list]]] = None,
) -> dataframe.DataFrame:
    """
    Runs prediction on a BigQuery ML model.

    See the `BigQuery ML PREDICT function syntax
    <https://docs.cloud.google.com/bigquery/docs/reference/standard-sql/bigqueryml-syntax-predict>`_
    for additional reference.

    Args:
        model (bigframes.ml.base.BaseEstimator or str):
            The model to use for prediction.
        input_ (Union[bigframes.pandas.DataFrame, str]):
            The DataFrame or query to use for prediction.
        options (Mapping[str, Union[str, int, float, bool, list]], optional):
            The OPTIONS clause, which specifies the model options.

    Returns:
        bigframes.pandas.DataFrame:
            The prediction results.
    """
    model_name, session = _get_model_name_and_session(model, input_)
    table_sql = _to_sql(input_)

    sql = bigframes.core.sql.ml.predict(
        model_name=model_name,
        table=table_sql,
        options=options,
    )

    return session.read_gbq(sql)


@log_adapter.method_logger(custom_base_name="bigquery_ml")
def explain_predict(
    model: Union[bigframes.ml.base.BaseEstimator, str],
    input_: Union[dataframe.DataFrame, str],
    *,
    options: Optional[Mapping[str, Union[str, int, float, bool, list]]] = None,
) -> dataframe.DataFrame:
    """
    Runs explainable prediction on a BigQuery ML model.

    See the `BigQuery ML EXPLAIN_PREDICT function syntax
    <https://docs.cloud.google.com/bigquery/docs/reference/standard-sql/bigqueryml-syntax-explain-predict>`_
    for additional reference.

    Args:
        model (bigframes.ml.base.BaseEstimator or str):
            The model to use for prediction.
        input_ (Union[bigframes.pandas.DataFrame, str]):
            The DataFrame or query to use for prediction.
        options (Mapping[str, Union[str, int, float, bool, list]], optional):
            The OPTIONS clause, which specifies the model options.

    Returns:
        bigframes.pandas.DataFrame:
            The prediction results with explanations.
    """
    model_name, session = _get_model_name_and_session(model, input_)
    table_sql = _to_sql(input_)

    sql = bigframes.core.sql.ml.explain_predict(
        model_name=model_name,
        table=table_sql,
        options=options,
    )

    return session.read_gbq(sql)


@log_adapter.method_logger(custom_base_name="bigquery_ml")
def global_explain(
    model: Union[bigframes.ml.base.BaseEstimator, str],
    *,
    options: Optional[Mapping[str, Union[str, int, float, bool, list]]] = None,
) -> dataframe.DataFrame:
    """
    Gets global explanations for a BigQuery ML model.

    See the `BigQuery ML GLOBAL_EXPLAIN function syntax
    <https://docs.cloud.google.com/bigquery/docs/reference/standard-sql/bigqueryml-syntax-global-explain>`_
    for additional reference.

    Args:
        model (bigframes.ml.base.BaseEstimator or str):
            The model to get explanations from.
        options (Mapping[str, Union[str, int, float, bool, list]], optional):
            The OPTIONS clause, which specifies the model options.

    Returns:
        bigframes.pandas.DataFrame:
            The global explanation results.
    """
    model_name, session = _get_model_name_and_session(model)
    sql = bigframes.core.sql.ml.global_explain(
        model_name=model_name,
        options=options,
    )

    return session.read_gbq(sql)
