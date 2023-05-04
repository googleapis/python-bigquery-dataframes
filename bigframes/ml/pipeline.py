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

"""Implements Scikit-Learn's sklearn.pipeline API"""

from __future__ import annotations

import typing
from typing import List, Optional, Tuple

import bigframes
import bigframes.ml.api_primitives
import bigframes.ml.cluster
import bigframes.ml.core
import bigframes.ml.linear_model
import bigframes.ml.preprocessing
import bigframes.ml.sql


class Pipeline:
    """A pipeline of transforms with a final estimator

    This allows chaining preprocessing steps onto an estimator to produce
    a single component. This simplifies code, and allows deploying an estimator
    and peprocessing together, e.g. with Pipeline.to_gbq(...)"""

    def __init__(
        self, steps: List[Tuple[str, bigframes.ml.api_primitives.BaseEstimator]]
    ):
        """Parameters:
        steps: a list of tuples of (name, estimator). Every estimator but the final
            one must implement a .transform(...) method. The final estimator need
            only implement .fit(...)"""
        self._steps = steps
        self._bqml_model: Optional[bigframes.ml.core.BqmlModel] = None

    def _compile_transforms(self, input_schema: List[str]) -> List[str]:
        """Combine transform steps with the schema of the input data to produce a list of SQL
        expressions for the TRANSFORM clause."""
        # TODO(bmil): input schema should have types also & be validated
        # TODO(bmil): handle multiple transform types
        # TODO(bmil): handle ColumnTransformer (& dedupe output names with input schema)

        # For now only a single StandardScaler step supported
        if len(self._steps) != 2:
            raise NotImplementedError(
                "Currently only two step (transform, estimator) pipelines are supported"
            )

        _, transform = self._steps[0]
        if isinstance(transform, bigframes.ml.preprocessing.StandardScaler):
            return [
                bigframes.ml.sql.ml_standard_scaler(column, f"scaled_{column}")
                for column in input_schema
            ]
        else:
            raise NotImplementedError(
                f"Transform {transform} is not yet supported by Pipeline"
            )

    def fit(self, X: bigframes.DataFrame, y: Optional[bigframes.DataFrame] = None):
        """Fit each estimator in the pipeline to the transformed output of the
        previous one. In bigframes.ml this will compile the pipeline to a single
        BQML model with a TRANSFORM clause

        Parameters:
            X: training data. Must match the input requirements of the first step of
                the pipeline
            y: training targets, if applicable"""
        # TODO(bmil): determine use cases for longer chains
        # for now, lets just support two step pipelines
        if len(self._steps) != 2:
            raise NotImplementedError(
                "Currently only two step (transform, estimator) pipelines are supported"
            )

        # TODO(bmil): BigFrames dataframe<->SQL mapping is being reworked, this will need to be updated
        # TODO(bmil): It may be tidier to have standard ways for transforms to implement parts of their
        # compilation to SQL, but for now, lets do everything in Pipeline
        transform_sql_exprs = self._compile_transforms(X.columns.tolist())

        # If labels columns are present, they should pass through un-transformed
        if y is not None:
            transform_sql_exprs.extend(y.columns.tolist())

        _, estimator = self._steps[1]
        # TODO(bmil): add a common mixin class for Predictors, check against that instead
        if not isinstance(
            estimator,
            (bigframes.ml.linear_model.LinearRegression, bigframes.ml.cluster.KMeans),
        ):
            raise NotImplementedError(
                f"Estimator {estimator} is currently not supported by Pipeline"
            )

        self._bqml_model = bigframes.ml.core.create_bqml_model(
            train_X=X,
            train_y=y,
            transforms=transform_sql_exprs,
            options=estimator._bqml_options,
        )

    def predict(self, X: bigframes.DataFrame) -> bigframes.DataFrame:
        if not self._bqml_model:
            raise RuntimeError("A pipeline must be fitted before predict")

        # TODO(bmil): This implementation only works for supervised models. Need to rework it.
        # Potentially we should delegate more functionality to the estimator, and have pipeline
        # pass in the transforms.
        df = self._bqml_model.predict(X)
        return typing.cast(
            bigframes.dataframe.DataFrame,
            df[
                [
                    typing.cast(str, field.name)
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
            raise RuntimeError("A pipeline must be fitted before score")

        if (X is None) != (y is None):
            raise ValueError(
                "Either both or neither of test_X and test_y must be specified"
            )
        input_data = X.join(y, how="outer") if X and y else None
        return self._bqml_model.evaluate(input_data)

    def to_gbq(self, model_name: str, replace: bool = False) -> Pipeline:
        if not self._bqml_model:
            raise RuntimeError("A pipeline must be fitted before it can be saved")

        new_model = self._bqml_model.copy(model_name, replace)
        return new_model.session.read_gbq_model(self._bqml_model.model_name)
