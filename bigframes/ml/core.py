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

"""Core operations for BQML based models"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Union
import uuid

from google.cloud import bigquery

import bigframes.dataframe
import bigframes.ml.sql
import bigframes.session


class BqmlModel:
    """Represents an existing BQML model in BigQuery
    Wraps the BQML API and SQL interface to expose the functionality needed for BigFrames ML"""

    def __init__(self, session: bigframes.session.Session, model: bigquery.Model):
        self._session = session
        self._model = model

    @property
    def session(self) -> bigframes.Session:
        """Get the BigFrames session that this BQML model wrapper is tied to"""
        return self._session

    @property
    def model_name(self):
        """Get the fully qualified name of the model, i.e. project_id.dataset_id.model_id"""
        return f"{self._model.project}.{self._model.dataset_id}.{self._model.model_id}"

    @property
    def model(self) -> bigquery.Model:
        """Get the BQML model associated with this wrapper"""
        return self._model

    def predict(
        self, input_data: bigframes.dataframe.DataFrame
    ) -> bigframes.dataframe.DataFrame:
        # TODO: validate input data schema
        sql = bigframes.ml.sql.ml_predict(
            model_name=self.model_name, source_sql=input_data.sql
        )

        # TODO: implement a more robust way to get index columns from a dataframe's SQL
        index_columns = input_data._block.index_columns
        df = self._session.read_gbq(sql, index_cols=index_columns)

        return df

    def evaluate(self, input_data: Union[bigframes.dataframe.DataFrame, None] = None):
        # TODO: validate input data schema
        sql = bigframes.ml.sql.ml_evaluate(
            self.model_name, input_data.sql if input_data else None
        )
        return self._session.read_gbq(sql)

    def copy(self, new_model_name, replace=False) -> BqmlModel:
        job_config = bigquery.job.CopyJobConfig()
        if replace:
            job_config.write_disposition = "WRITE_TRUNCATE"

        self._session.bqclient.copy_table(
            self.model_name, new_model_name, job_config=job_config
        ).result()

        new_model = self._session.bqclient.get_model(new_model_name)
        return BqmlModel(self._session, new_model)


def create_bqml_model(
    train_X: bigframes.dataframe.DataFrame,
    train_y: Optional[bigframes.dataframe.DataFrame] = None,
    transforms: Optional[Iterable[str]] = None,
    options: Dict[str, Union[str, int, float, List[str]]] = {},
) -> BqmlModel:
    """Create a session-temporary BQML model with the CREATE MODEL statement

    Parameters:
        train_X: features columns for training
        train_y: labels columns for training, if applicable
        transforms: an optional list of SQL expressions that implement preprocessing
            on top of the input data. Generates a BQML TRANSFORM clause
        options: a dict of options to configure the model. Generates a BQML OPTIONS
            clause

    Returns: a BqmlModel, wrapping a trained model in BigQuery
    """
    if train_y is None:
        input_data = train_X
    else:
        # TODO: handle case where train_y columns are renamed in the join
        input_data = train_X.join(train_y, how="outer")
        options.update({"input_label_cols": train_y.columns.tolist()})

    # pickpocket session object from the dataframe
    session = train_X._block.expr._session

    # TODO(bmil): add wrapper to select the feature columns
    # for now, drop index to avoid including the index in feature columns
    input_data = input_data.reset_index(drop=True)

    model_name = f"{session._session_dataset_id}.{uuid.uuid4().hex}"
    source_sql = input_data.sql
    options_sql = bigframes.ml.sql.options(**options)
    transform_sql = (
        bigframes.ml.sql.transform(*transforms) if transforms is not None else None
    )
    sql = bigframes.ml.sql.create_model(
        model_name=model_name,
        source_sql=source_sql,
        transform_sql=transform_sql,
        options_sql=options_sql,
    )

    # fit the model, synchronously
    session.bqclient.query(sql).result()

    model = session.bqclient.get_model(model_name)
    return BqmlModel(session, model)
