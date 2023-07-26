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

from __future__ import annotations

from typing import cast, Union

import bigframes
from bigframes.ml import base, core, utils
import bigframes.pandas as bpd


class TensorFlowModel(base.Predictor):
    """Imported TensorFlow model.

    Args:
        session (BigQuery Session):
            BQ session to create the model
        model_path (str):
            GCS path that holds the model files."""

    def __init__(self, session: bigframes.Session, model_path: str):
        self.session = session
        self.model_path = model_path
        self._bqml_model: core.BqmlModel = self._create_bqml_model()

    def _create_bqml_model(self):
        options = {"model_type": "TENSORFLOW", "model_path": self.model_path}
        return core.create_bqml_imported_model(session=self.session, options=options)

    def predict(self, X: Union[bpd.DataFrame, bpd.Series]) -> bpd.DataFrame:
        """Predict the result from input DataFrame.

        Args:
            X (BigQuery DataFrame):
                Input DataFrame, schema is defined by the model.

        Returns:
            BigQuery DataFrame: Output DataFrame, schema is defined by the model."""
        (X,) = utils.convert_to_dataframe(X)

        df = self._bqml_model.predict(X)
        return cast(
            bpd.DataFrame,
            df[
                [
                    cast(str, field.name)
                    for field in self._bqml_model.model.label_columns
                ]
            ],
        )


class ONNXModel(base.Predictor):
    """Imported Open Neural Network Exchange (ONNX) model.

    Args:
        session (BigQuery Session):
            BQ session to create the model
        model_path (str):
            GCS path that holds the model files."""

    def __init__(self, session: bigframes.Session, model_path: str):
        self.session = session
        self.model_path = model_path
        self._bqml_model: core.BqmlModel = self._create_bqml_model()

    def _create_bqml_model(self):
        options = {"model_type": "ONNX", "model_path": self.model_path}
        return core.create_bqml_imported_model(session=self.session, options=options)

    def predict(self, X: Union[bpd.DataFrame, bpd.Series]) -> bpd.DataFrame:
        """Predict the result from input DataFrame.

        Args:
            X (BigQuery DataFrame or Series):
                Input DataFrame or Series, schema is defined by the model.

        Returns:
            BigQuery DataFrame: Output DataFrame, schema is defined by the model."""
        (X,) = utils.convert_to_dataframe(X)

        df = self._bqml_model.predict(X)
        return cast(
            bpd.DataFrame,
            df[
                [
                    cast(str, field.name)
                    for field in self._bqml_model.model.label_columns
                ]
            ],
        )
