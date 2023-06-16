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

"""For composing estimators together. This module is styled after Scikit-Learn's
pipeline module: https://scikit-learn.org/stable/modules/pipeline.html"""


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


class Pipeline(bigframes.ml.api_primitives.BaseEstimator):
    """A pipeline of transforms with a final estimator

    This allows chaining preprocessing steps onto an estimator to produce
    a single component. This simplifies code, and allows deploying an estimator
    and peprocessing together, e.g. with Pipeline.to_gbq(...)

    Currently in bigframes.ml only two step pipelines are supported. The first
    step should be preprocessing, and the second step should be the model."""

    def __init__(
        self, steps: List[Tuple[str, bigframes.ml.api_primitives.BaseEstimator]]
    ):
        self.steps = steps

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

        self._transform = transform
        self._estimator = estimator

    def fit(self, X: bigframes.DataFrame, y: Optional[bigframes.DataFrame] = None):
        """Fit each estimator in the pipeline to the transformed output of the
        previous one. In bigframes.ml this will compile the pipeline to a single
        BQML model with a TRANSFORM clause

        Args:
            X: training data. Must match the input requirements of the first step of
                the pipeline
            y: training targets, if applicable"""

        compiled_transforms = self._transform._compile_to_sql(X.columns.tolist())
        transform_sqls = [transform_sql for transform_sql, _ in compiled_transforms]

        # TODO(bmil): need a more elegant way to address this. Perhaps a mixin for supervised vs
        # unsupervised models? Or just lists of classes?
        if isinstance(self._estimator, bigframes.ml.cluster.KMeans):
            self._estimator.fit(X=X, transforms=transform_sqls)
        else:
            if y is not None:
                # If labels columns are present, they should pass through un-transformed
                transform_sqls.extend(y.columns.tolist())
                self._estimator.fit(X=X, y=y, transforms=transform_sqls)
            else:
                raise TypeError("Fitting this pipeline requires training targets `y`")

    def predict(self, X: bigframes.DataFrame) -> bigframes.DataFrame:
        """Predict the pipeline result for each sample in X.

        Args:
            X: a BigFrames DataFrame to predict.

        Return: a BigFrames Dataframe representing predicted result.
        """
        return self._estimator.predict(X)

    def score(
        self,
        X: Optional[bigframes.DataFrame] = None,
        y: Optional[bigframes.DataFrame] = None,
    ):
        """Calculate evaluation metrics of the model.

        Args:
            X: a BigFrames DataFrame as evaluation data.
            y: a BigFrames DataFrame as evaluation labels.

        Returns: a BigFrames DataFrame as evaluation result."""
        if isinstance(self._estimator, bigframes.ml.linear_model.LinearRegression):
            return self._estimator.score(X=X, y=y)

    def to_gbq(self, model_name: str, replace: bool = False):
        self._estimator.to_gbq(model_name, replace)

        # TODO: should instead load from GBQ, but loading pipelines is not implemented yet
        return self
