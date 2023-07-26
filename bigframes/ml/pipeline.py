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

from typing import List, Optional, Tuple, Union

from bigframes.ml import base, compose, preprocessing, utils
import bigframes.pandas as bpd
import third_party.bigframes_vendored.sklearn.pipeline


class Pipeline(
    third_party.bigframes_vendored.sklearn.pipeline.Pipeline,
    base.BaseEstimator,
):
    __doc__ = third_party.bigframes_vendored.sklearn.pipeline.Pipeline.__doc__

    def __init__(self, steps: List[Tuple[str, base.BaseEstimator]]):
        self.steps = steps

        if len(steps) != 2:
            raise NotImplementedError(
                "Currently only two step (transform, estimator) pipelines are supported"
            )

        transform, estimator = steps[0][1], steps[1][1]
        if isinstance(
            transform,
            (
                compose.ColumnTransformer,
                preprocessing.StandardScaler,
                preprocessing.OneHotEncoder,
            ),
        ):
            self._transform = transform
        else:
            raise NotImplementedError(
                f"Transform {transform} is not yet supported by Pipeline"
            )

        if not isinstance(
            estimator,
            base.TrainablePredictor,
        ):
            raise NotImplementedError(
                f"Estimator {estimator} is not supported by Pipeline"
            )

        self._transform = transform
        self._estimator = estimator

    def fit(
        self,
        X: Union[bpd.DataFrame, bpd.Series],
        y: Optional[Union[bpd.DataFrame, bpd.Series]] = None,
    ):
        (X,) = utils.convert_to_dataframe(X)

        compiled_transforms = self._transform._compile_to_sql(X.columns.tolist())
        transform_sqls = [transform_sql for transform_sql, _ in compiled_transforms]

        if y is not None:
            # If labels columns are present, they should pass through un-transformed
            (y,) = utils.convert_to_dataframe(y)
            transform_sqls.extend(y.columns.tolist())

        self._estimator.fit(X=X, y=y, transforms=transform_sqls)

    def predict(self, X: Union[bpd.DataFrame, bpd.Series]) -> bpd.DataFrame:
        return self._estimator.predict(X)

    def score(
        self,
        X: Union[bpd.DataFrame, bpd.Series],
        y: Optional[Union[bpd.DataFrame, bpd.Series]] = None,
    ) -> bpd.DataFrame:
        (X,) = utils.convert_to_dataframe(X)
        if y is not None:
            (y,) = utils.convert_to_dataframe(y)

        return self._estimator.score(X=X, y=y)

    def to_gbq(self, model_name: str, replace: bool = False):
        self._estimator.to_gbq(model_name, replace)

        # TODO: should instead load from GBQ, but loading pipelines is not implemented yet
        return self
