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

"""Linear models. This module is styled after Scikit-Learn's linear_model module:
https://scikit-learn.org/stable/modules/linear_model.html"""

from __future__ import annotations

from typing import cast, Dict, List, Literal, Optional, TYPE_CHECKING

from google.cloud import bigquery

if TYPE_CHECKING:
    import bigframes

import bigframes.ml.api_primitives
import bigframes.ml.core


class XGBRegressor(bigframes.ml.api_primitives.BaseEstimator):
    """Boosted Tree Regressor model.

    Args:
        num_parallel_tree: number of parallel trees constructed during each iteration. Default to 1.
        booster_type: specify the booster type to use. Default to "GBTREE".
        early_stop: whether training should stop after the first iteration. Default to True.
        data_split_method: whether to auto split data. Possible values: "NO_SPLIT", "AUTO_SPLIT". Default to "NO_SPLIT".
        subsample: subsample ratio of the training instances. Default to 1.0.
    """

    def __init__(
        self,
        num_parallel_tree=1,
        booster_type: Literal["GBTREE", "DART"] = "GBTREE",
        early_stop=True,
        data_split_method: Literal["NO_SPLIT", "AUTO_SPLIT"] = "NO_SPLIT",
        subsample=1.0,
    ):
        self.num_parallel_tree = num_parallel_tree
        self.booster_type = booster_type
        self.early_stop = early_stop
        self.data_split_method = data_split_method
        self.subsample = subsample
        self._bqml_model: Optional[bigframes.ml.core.BqmlModel] = None

    @staticmethod
    def _from_bq(session: bigframes.Session, model: bigquery.Model) -> XGBRegressor:
        assert model.model_type == "BOOSTED_TREE_REGRESSOR"

        kwargs = {}

        # See https://cloud.google.com/bigquery/docs/reference/rest/v2/models#trainingrun
        last_fitting = model.training_runs[-1]["trainingOptions"]
        if "boosterType" in last_fitting:
            kwargs["booster_type"] = last_fitting["boosterType"]
        if "earlyStop" in last_fitting:
            kwargs["early_stop"] = last_fitting["earlyStop"]
        if "dataSplitMethod" in last_fitting:
            kwargs["data_split_method"] = last_fitting["dataSplitMethod"]
        if "subsample" in last_fitting:
            kwargs["subsample"] = float(last_fitting["subsample"])
        if "numParallelTree" in last_fitting:
            kwargs["num_parallel_tree"] = int(last_fitting["numParallelTree"])

        new_xgb_regressor = XGBRegressor(**kwargs)
        new_xgb_regressor._bqml_model = bigframes.ml.core.BqmlModel(session, model)
        return new_xgb_regressor

    @property
    def _bqml_options(self) -> Dict[str, str | int | bool | float | List[str]]:
        """The model options as they will be set for BQML"""
        return {
            "model_type": "BOOSTED_TREE_REGRESSOR",
            "num_parallel_tree": self.num_parallel_tree,
            "booster_type": self.booster_type,
            "early_stop": self.early_stop,
            "data_split_method": self.data_split_method,
            "subsample": self.subsample,
        }

    def fit(
        self,
        X: bigframes.DataFrame,
        y: bigframes.DataFrame,
    ):
        """Fit the model to training data

        Args:
            X: A dataframe with training data

            y: Target values for training
        """
        self._bqml_model = bigframes.ml.core.create_bqml_model(
            X,
            y,
            options=self._bqml_options,
        )

    def predict(self, X: bigframes.DataFrame) -> bigframes.DataFrame:
        """Predict the closest cluster for each sample in X.

        Args:
            X: a BigFrames DataFrame to predict.

        Returns: predicted BigFrames DataFrame."""
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
        """Calculate evaluation metrics of the model.

        Args:
            X: a BigFrames DataFrame as evaluation data.
            y: a BigFrames DataFrame as evaluation labels.

        Returns: a BigFrames DataFrame as evaluation result."""
        if not self._bqml_model:
            raise RuntimeError("A model must be fitted before score")

        if (X is None) != (y is None):
            raise ValueError(
                "Either both or neither of test_X and test_y must be specified"
            )
        input_data = X.join(y, how="outer") if X and y else None
        return self._bqml_model.evaluate(input_data)

    def to_gbq(self, model_name: str, replace: bool = False) -> XGBRegressor:
        """Save the model to Google Cloud BigQuey.

        Args:
            model_name: the name of the model.
            replace: whether to replace if the model already exists. Default to False.

        Returns: saved model."""
        if not self._bqml_model:
            raise RuntimeError("A model must be fitted before it can be saved")

        new_model = self._bqml_model.copy(model_name, replace)
        return new_model.session.read_gbq_model(model_name)


class XGBClassifier(bigframes.ml.api_primitives.BaseEstimator):
    """Boosted Tree Classifier model.

    Args:
        num_parallel_tree: number of parallel trees constructed during each iteration. Default to 1.
        booster_type: specify the booster type to use. Default to "GBTREE".
        early_stop: whether training should stop after the first iteration. Default to True.
        data_split_method: whether to auto split data. Possible values: "NO_SPLIT", "AUTO_SPLIT". Default to "NO_SPLIT".
        subsample: subsample ratio of the training instances. Default to 1.0.
    """

    def __init__(
        self,
        num_parallel_tree=1,
        booster_type: Literal["GBTREE", "DART"] = "GBTREE",
        early_stop=True,
        data_split_method: Literal["NO_SPLIT", "AUTO_SPLIT"] = "NO_SPLIT",
        subsample=1.0,
    ):
        self.num_parallel_tree = num_parallel_tree
        self.booster_type = booster_type
        self.early_stop = early_stop
        self.data_split_method = data_split_method
        self.subsample = subsample
        self._bqml_model: Optional[bigframes.ml.core.BqmlModel] = None

    @staticmethod
    def _from_bq(session: bigframes.Session, model: bigquery.Model) -> XGBClassifier:
        assert model.model_type == "BOOSTED_TREE_CLASSIFIER"

        kwargs = {}

        # See https://cloud.google.com/bigquery/docs/reference/rest/v2/models#trainingrun
        last_fitting = model.training_runs[-1]["trainingOptions"]
        if "boosterType" in last_fitting:
            kwargs["booster_type"] = last_fitting["boosterType"]
        if "earlyStop" in last_fitting:
            kwargs["early_stop"] = last_fitting["earlyStop"]
        if "dataSplitMethod" in last_fitting:
            kwargs["data_split_method"] = last_fitting["dataSplitMethod"]
        if "subsample" in last_fitting:
            kwargs["subsample"] = float(last_fitting["subsample"])
        if "numParallelTree" in last_fitting:
            kwargs["num_parallel_tree"] = int(last_fitting["numParallelTree"])

        new_xgb_classifier = XGBClassifier(**kwargs)
        new_xgb_classifier._bqml_model = bigframes.ml.core.BqmlModel(session, model)
        return new_xgb_classifier

    @property
    def _bqml_options(self) -> Dict[str, str | int | bool | float | List[str]]:
        """The model options as they will be set for BQML"""
        return {
            "model_type": "BOOSTED_TREE_CLASSIFIER",
            "num_parallel_tree": self.num_parallel_tree,
            "booster_type": self.booster_type,
            "early_stop": self.early_stop,
            "data_split_method": self.data_split_method,
            "subsample": self.subsample,
        }

    def fit(
        self,
        X: bigframes.DataFrame,
        y: bigframes.DataFrame,
    ):
        """Fit the model to training data

        Args:
            X: A dataframe with training data

            y: Target values for training
        """
        self._bqml_model = bigframes.ml.core.create_bqml_model(
            X,
            y,
            options=self._bqml_options,
        )

    def predict(self, X: bigframes.DataFrame) -> bigframes.DataFrame:
        """Predict the closest cluster for each sample in X.

        Args:
            X: a BigFrames DataFrame to predict.

        Returns: predicted BigFrames DataFrame."""
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
        """Calculate evaluation metrics of the model.

        Args:
            X: a BigFrames DataFrame as evaluation data.
            y: a BigFrames DataFrame as evaluation labels.

        Returns: a BigFrames DataFrame as evaluation result."""
        if not self._bqml_model:
            raise RuntimeError("A model must be fitted before score")

        if (X is None) != (y is None):
            raise ValueError(
                "Either both or neither of test_X and test_y must be specified"
            )
        input_data = X.join(y, how="outer") if X and y else None
        return self._bqml_model.evaluate(input_data)

    def to_gbq(self, model_name: str, replace: bool = False) -> XGBClassifier:
        """Save the model to Google Cloud BigQuey.

        Args:
            model_name: the name of the model.
            replace: whether to replace if the model already exists. Default to False.

        Returns: saved model."""
        if not self._bqml_model:
            raise RuntimeError("A model must be fitted before it can be saved")

        new_model = self._bqml_model.copy(model_name, replace)
        return new_model.session.read_gbq_model(model_name)
