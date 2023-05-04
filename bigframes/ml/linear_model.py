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

from typing import cast, Dict, List, Optional, TYPE_CHECKING

from google.cloud import bigquery

if TYPE_CHECKING:
    import bigframes

import bigframes.ml.api_primitives
import bigframes.ml.core


class LinearRegression(bigframes.ml.api_primitives.BaseEstimator):
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self._bqml_model: Optional[bigframes.ml.core.BqmlModel] = None

    @staticmethod
    def _from_bq(session: bigframes.Session, model: bigquery.Model) -> LinearRegression:
        assert model.model_type == "LINEAR_REGRESSION"

        # TODO(bmil): construct a standard way to extract these properties
        kwargs = {}
        last_fitting = model.training_runs[-1]
        if "fitIntercept" in last_fitting:
            kwargs["last_fitting"] = last_fitting["fitIntercept"]

        new_linear_regression = LinearRegression(**kwargs)
        new_linear_regression._bqml_model = bigframes.ml.core.BqmlModel(session, model)
        return new_linear_regression

    @property
    def _bqml_options(self) -> Dict[str, str | int | float | List[str]]:
        """The model options as they will be set for BQML"""
        return {"model_type": "LINEAR_REG", "fit_intercept": self.fit_intercept}

    def fit(self, X: bigframes.DataFrame, y: bigframes.DataFrame):
        self._bqml_model = bigframes.ml.core.create_bqml_model(
            X,
            y,
            options=self._bqml_options,
        )

    def predict(self, X: bigframes.DataFrame) -> bigframes.DataFrame:
        if not self._bqml_model:
            raise RuntimeError("A model must be fitted before predict")

        df = self._bqml_model.predict(X)
        return cast(
            bigframes.dataframe.DataFrame,
            df[
                [
                    cast(str, field.name)
                    for field in self._bqml_model.model.label_columns
                ]
            ],
        )

    def score(
        self,
        X: Optional[bigframes.DataFrame] = None,
        y: Optional[bigframes.DataFrame] = None,
    ):
        if not self._bqml_model:
            raise RuntimeError("A model must be fitted before score")

        if (X is None) != (y is None):
            raise ValueError(
                "Either both or neither of test_X and test_y must be specified"
            )
        input_data = X.join(y, how="outer") if X and y else None
        return self._bqml_model.evaluate(input_data)

    def to_gbq(self, model_name: str, replace: bool = False) -> LinearRegression:
        if not self._bqml_model:
            raise RuntimeError("A model must be fitted before it can be saved")

        new_model = self._bqml_model.copy(model_name, replace)
        return new_model.session.read_gbq_model(self._bqml_model.model_name)
