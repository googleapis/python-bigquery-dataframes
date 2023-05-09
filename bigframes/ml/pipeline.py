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

from typing import List, Optional, Tuple

import bigframes
import bigframes.ml.api_primitives
import bigframes.ml.cluster
import bigframes.ml.compose
import bigframes.ml.core
import bigframes.ml.linear_model
import bigframes.ml.preprocessing
import bigframes.ml.sql


class Pipeline:
    """A pipeline of transforms with a final estimator

    This allows chaining preprocessing steps onto an estimator to produce
    a single component. This simplifies code, and allows deploying an estimator
    and peprocessing together, e.g. with Pipeline.to_gbq(...)

    Currently in bigframes.ml only two step pipelines are supported. The first
    step should be preprocessing, and the second step should be the model."""

    def __init__(
        self, steps: List[Tuple[str, bigframes.ml.api_primitives.BaseEstimator]]
    ):
        if len(steps) != 2:
            raise NotImplementedError(
                "Currently only two step (transform, estimator) pipelines are supported"
            )

        transform, estimator = steps[0][1], steps[1][1]
        if isinstance(
            transform,
            (
                bigframes.ml.compose.ColumnTransformer,
                bigframes.ml.preprocessing.StandardScaler,
                bigframes.ml.preprocessing.OneHotEncoder,
            ),
        ):
            self._transform = transform
        else:
            raise NotImplementedError(
                f"Transform {transform} is not yet supported by Pipeline"
            )

        if isinstance(
            estimator,
            (bigframes.ml.linear_model.LinearRegression, bigframes.ml.cluster.KMeans),
        ):
            self._estimator = estimator
        else:
            raise NotImplementedError(
                f"Estimator {estimator} is not yet supported by Pipeline"
            )

        self._steps = steps
        self._transform = transform
        self._estimator = estimator

    def _compile_transforms(self, input_schema: List[str]) -> List[str]:
        """Combine transform steps with the schema of the input data to produce a list of SQL
        expressions for the TRANSFORM clause."""
        # TODO(bmil): input schema should have types also & be validated
        # TODO(bmil): add some more graceful abstraction for representing preprocessors
        def apply_preprocessor(
            column: str, preprocessor: bigframes.ml.preprocessing.PreprocessorType
        ) -> str:
            if isinstance(preprocessor, bigframes.ml.preprocessing.StandardScaler):
                return bigframes.ml.sql.ml_standard_scaler(column, f"scaled_{column}")
            elif isinstance(preprocessor, bigframes.ml.preprocessing.OneHotEncoder):
                return bigframes.ml.sql.ml_one_hot_encoder(
                    column, f"one_hot_encoded_{column}"
                )

        transform_exprs: List[str] = []
        for column in input_schema:
            if isinstance(
                self._transform,
                (
                    bigframes.ml.preprocessing.StandardScaler,
                    bigframes.ml.preprocessing.OneHotEncoder,
                ),
            ):
                transform_exprs.append(apply_preprocessor(column, self._transform))
            else:  # ColumnTransformer
                for entry in self._transform.transformers_:
                    _, preprocessor, target_column = entry
                    if column == target_column:
                        transform_exprs.append(apply_preprocessor(column, preprocessor))

        return transform_exprs

    def fit(self, X: bigframes.DataFrame, y: Optional[bigframes.DataFrame] = None):
        """Fit each estimator in the pipeline to the transformed output of the
        previous one. In bigframes.ml this will compile the pipeline to a single
        BQML model with a TRANSFORM clause

        Args:
            X: training data. Must match the input requirements of the first step of
                the pipeline
            y: training targets, if applicable"""
        # TODO(bmil): BigFrames dataframe<->SQL mapping is being reworked, this will need to be updated
        # TODO(bmil): It may be tidier to have standard ways for transforms to implement parts of their
        # compilation to SQL, but for now, lets do everything in Pipeline
        transforms = self._compile_transforms(X.columns.tolist())

        # TODO(bmil): need a more elegant way to address this. Perhaps a mixin for supervised vs
        # unsupervised models? Or just lists of classes?
        if isinstance(self._estimator, bigframes.ml.cluster.KMeans):
            self._estimator.fit(X=X, transforms=transforms)
        else:
            if y is not None:
                # If labels columns are present, they should pass through un-transformed
                transforms.extend(y.columns.tolist())
                self._estimator.fit(X=X, y=y, transforms=transforms)
            else:
                raise TypeError("Fitting this pipeline requires training targets `y`")

    def predict(self, X: bigframes.DataFrame) -> bigframes.DataFrame:
        return self._estimator.predict(X)

    def score(
        self,
        X: Optional[bigframes.DataFrame] = None,
        y: Optional[bigframes.DataFrame] = None,
    ):
        if isinstance(self._estimator, bigframes.ml.linear_model.LinearRegression):
            return self._estimator.score(X=X, y=y)
