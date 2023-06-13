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

_BQML_PARAMS_MAPPING = {
    "booster": "boosterType",
    "tree_method": "treeMethod",
    "early_stop": "earlyStop",
    "data_split_method": "dataSplitMethod",
    "colsample_bytree": "colsampleBylevel",
    "colsample_bylevel": "colsampleBytree",
    "colsample_bynode": "colsampleBynode",
    "gamma": "minSplitLoss",
    "subsample": "subsample",
    "reg_alpha": "l1Regularization",
    "reg_lambda": "l2Regularization",
    "learning_rate": "learnRate",
    "min_rel_progress": "minRelativeProgress",
    "num_parallel_tree": "numParallelTree",
    "min_tree_child_weight": "minTreeChildWeight",
    "max_depth": "maxTreeDepth",
    "max_iterations": "maxIterations",
}


class XGBRegressor(bigframes.ml.api_primitives.BaseEstimator):
    """Boosted Tree Regressor model.

    Args:
        num_parallel_tree: number of parallel trees constructed during each iteration. Default to 1.
        booster: specify the booster type to use. Possible values: "GBTREE", "DART". Default to "GBTREE".
        dart_normalized_type: type of normalization algorithm for DART booster. Possible values: "TREE", "FOREST". Default to "TREE".
        tree_method: type of tree construction algorithm. Possible values: "AUTO", "EXACT", "APPROX", "HIST". Default value to "AUTO".
        min_tree_child_weight: minimum sum of instance weight needed in a child for further partitioning. The value should be greater than or equal to 0. Default to 1.
        colsample_bytree: subsample ratio of columns when constructing each tree. The value should be between 0 and. Default to 1.0.
        colsample_bylevel: subsample ratio of columns for each level. The value should be between 0 and. Default to 1.0.
        colsample_bynode: subsample ratio of columns for each node (split). The value should be between 0 and. Default to 1.0.
        gamma: minimum loss reduction required to make a further partition on a leaf node of the tree. Default to 0.0.
        max_depth: maximum depth of a tree. Default to 6.
        subsample: subsample ratio of the training instances. Default to 1.0.
        reg_alpha: the amount of L1 regularization applied. Default to 0.0.
        reg_lambda: the amount of L2 regularization applied. Default to 1.0.
        early_stop: whether training should stop after the first iteration. Default to True.
        learning_rate: step size shrinkage used in update to prevents overfitting. Default to 0.3.
        max_iterations: maximum number of rounds for boosting. Default to 20.
        min_rel_progress: minimum relative loss improvement necessary to continue training when early_stop is set to True. Default to 0.01.
        data_split_method: whether to auto split data. Possible values: "NO_SPLIT", "AUTO_SPLIT". Default to "NO_SPLIT".
        enable_global_explain: whether to compute global explanations using explainable AI to evaluate global feature importance to the model. Default to False.
        xgboost_version: specifies the Xgboost version for model training. Default to "0.9".
    """

    def __init__(
        self,
        num_parallel_tree: int = 1,
        booster: Literal["gbtree", "dart"] = "gbtree",
        dart_normalized_type: Literal["TREE", "FOREST"] = "TREE",
        tree_method: Literal["auto", "exact", "approx", "hist"] = "auto",
        min_tree_child_weight: int = 1,
        colsample_bytree=1.0,
        colsample_bylevel=1.0,
        colsample_bynode=1.0,
        gamma=0.0,
        max_depth: int = 6,
        subsample=1.0,
        reg_alpha=0.0,
        reg_lambda=1.0,
        early_stop=True,
        learning_rate=0.3,
        max_iterations: int = 20,
        min_rel_progress=0.01,
        data_split_method: Literal["NO_SPLIT", "AUTO_SPLIT"] = "NO_SPLIT",
        enable_global_explain=False,
        xgboost_version: Literal["0.9", "1.1"] = "0.9",
    ):
        self.num_parallel_tree = num_parallel_tree
        self.booster = booster
        self.dart_normalized_type = dart_normalized_type
        self.tree_method = tree_method
        self.min_tree_child_weight = min_tree_child_weight
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.colsample_bynode = colsample_bynode
        self.gamma = gamma
        self.max_depth = max_depth
        self.subsample = subsample
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.early_stop = early_stop
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.min_rel_progress = min_rel_progress
        self.data_split_method = data_split_method
        self.enable_global_explain = enable_global_explain
        self.xgboost_version = xgboost_version
        self._bqml_model: Optional[bigframes.ml.core.BqmlModel] = None

    @staticmethod
    def _from_bq(session: bigframes.Session, model: bigquery.Model) -> XGBRegressor:
        assert model.model_type == "BOOSTED_TREE_REGRESSOR"

        kwargs = {}

        # See https://cloud.google.com/bigquery/docs/reference/rest/v2/models#trainingrun
        last_fitting = model.training_runs[-1]["trainingOptions"]

        dummy_regressor = XGBRegressor()
        for bf_param, bf_value in dummy_regressor.__dict__.items():
            bqml_param = _BQML_PARAMS_MAPPING.get(bf_param)
            if bqml_param is not None:
                kwargs[bf_param] = type(bf_value)(last_fitting[bqml_param])

        new_xgb_regressor = XGBRegressor(**kwargs)
        new_xgb_regressor._bqml_model = bigframes.ml.core.BqmlModel(session, model)
        return new_xgb_regressor

    @property
    def _bqml_options(self) -> Dict[str, str | int | bool | float | List[str]]:
        """The model options as they will be set for BQML"""
        return {
            "model_type": "BOOSTED_TREE_REGRESSOR",
            "num_parallel_tree": self.num_parallel_tree,
            "booster_type": self.booster,
            "tree_method": self.tree_method,
            "min_tree_child_weight": self.min_tree_child_weight,
            "colsample_bytree": self.colsample_bytree,
            "colsample_bylevel": self.colsample_bylevel,
            "colsample_bynode": self.colsample_bynode,
            "min_split_loss": self.gamma,
            "max_tree_depth": self.max_depth,
            "subsample": self.subsample,
            "l1_reg": self.reg_alpha,
            "l2_reg": self.reg_lambda,
            "early_stop": self.early_stop,
            "learn_rate": self.learning_rate,
            "max_iterations": self.max_iterations,
            "min_rel_progress": self.min_rel_progress,
            "data_split_method": self.data_split_method,
            "enable_global_explain": self.enable_global_explain,
            "xgboost_version": self.xgboost_version,
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
        booster: specify the booster type to use. Possible values: "GBTREE", "DART". Default to "GBTREE".
        dart_normalized_type: type of normalization algorithm for DART booster. Possible values: "TREE", "FOREST". Default to "TREE".
        tree_method: type of tree construction algorithm. Possible values: "AUTO", "EXACT", "APPROX", "HIST". Default value to "AUTO".
        min_tree_child_weight: minimum sum of instance weight needed in a child for further partitioning. The value should be greater than or equal to 0. Default to 1.
        colsample_bytree: subsample ratio of columns when constructing each tree. The value should be between 0 and. Default to 1.0.
        colsample_bylevel: subsample ratio of columns for each level. The value should be between 0 and. Default to 1.0.
        colsample_bynode: subsample ratio of columns for each node (split). The value should be between 0 and. Default to 1.0.
        gamma: minimum loss reduction required to make a further partition on a leaf node of the tree. Default to 0.0.
        max_depth: maximum depth of a tree. Default to 6.
        subsample: subsample ratio of the training instances. Default to 1.0.
        reg_alpha: the amount of L1 regularization applied. Default to 0.0.
        reg_lambda: the amount of L2 regularization applied. Default to 1.0.
        early_stop: whether training should stop after the first iteration. Default to True.
        learning_rate: step size shrinkage used in update to prevents overfitting. Default to 0.3.
        max_iterations: maximum number of rounds for boosting. Default to 20.
        min_rel_progress: minimum relative loss improvement necessary to continue training when early_stop is set to True. Default to 0.01.
        data_split_method: whether to auto split data. Possible values: "NO_SPLIT", "AUTO_SPLIT". Default to "NO_SPLIT".
        enable_global_explain: whether to compute global explanations using explainable AI to evaluate global feature importance to the model. Default to False.
        xgboost_version: specifies the Xgboost version for model training. Default to "0.9".
    """

    def __init__(
        self,
        num_parallel_tree: int = 1,
        booster: Literal["gbtree", "dart"] = "gbtree",
        dart_normalized_type: Literal["TREE", "FOREST"] = "TREE",
        tree_method: Literal["auto", "exact", "approx", "hist"] = "auto",
        min_tree_child_weight: int = 1,
        colsample_bytree=1.0,
        colsample_bylevel=1.0,
        colsample_bynode=1.0,
        gamma=0.0,
        max_depth: int = 6,
        subsample=1.0,
        reg_alpha=0.0,
        reg_lambda=1.0,
        early_stop=True,
        learning_rate=0.3,
        max_iterations: int = 20,
        min_rel_progress=0.01,
        data_split_method: Literal["NO_SPLIT", "AUTO_SPLIT"] = "NO_SPLIT",
        enable_global_explain=False,
        xgboost_version: Literal["0.9", "1.1"] = "0.9",
    ):
        self.num_parallel_tree = num_parallel_tree
        self.booster = booster
        self.dart_normalized_type = dart_normalized_type
        self.tree_method = tree_method
        self.min_tree_child_weight = min_tree_child_weight
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.colsample_bynode = colsample_bynode
        self.gamma = gamma
        self.max_depth = max_depth
        self.subsample = subsample
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.early_stop = early_stop
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.min_rel_progress = min_rel_progress
        self.data_split_method = data_split_method
        self.enable_global_explain = enable_global_explain
        self.xgboost_version = xgboost_version
        self._bqml_model: Optional[bigframes.ml.core.BqmlModel] = None

    @staticmethod
    def _from_bq(session: bigframes.Session, model: bigquery.Model) -> XGBClassifier:
        assert model.model_type == "BOOSTED_TREE_CLASSIFIER"

        kwargs = {}

        # See https://cloud.google.com/bigquery/docs/reference/rest/v2/models#trainingrun
        last_fitting = model.training_runs[-1]["trainingOptions"]

        dummy_classifier = XGBClassifier()
        for bf_param, bf_value in dummy_classifier.__dict__.items():
            bqml_param = _BQML_PARAMS_MAPPING.get(bf_param)
            if bqml_param is not None:
                kwargs[bf_param] = type(bf_value)(last_fitting[bqml_param])

        new_xgb_classifier = XGBClassifier(**kwargs)
        new_xgb_classifier._bqml_model = bigframes.ml.core.BqmlModel(session, model)
        return new_xgb_classifier

    @property
    def _bqml_options(self) -> Dict[str, str | int | bool | float | List[str]]:
        """The model options as they will be set for BQML"""
        return {
            "model_type": "BOOSTED_TREE_CLASSIFIER",
            "num_parallel_tree": self.num_parallel_tree,
            "booster_type": self.booster,
            "tree_method": self.tree_method,
            "min_tree_child_weight": self.min_tree_child_weight,
            "colsample_bytree": self.colsample_bytree,
            "colsample_bylevel": self.colsample_bylevel,
            "colsample_bynode": self.colsample_bynode,
            "min_split_loss": self.gamma,
            "max_tree_depth": self.max_depth,
            "subsample": self.subsample,
            "l1_reg": self.reg_alpha,
            "l2_reg": self.reg_lambda,
            "early_stop": self.early_stop,
            "learn_rate": self.learning_rate,
            "max_iterations": self.max_iterations,
            "min_rel_progress": self.min_rel_progress,
            "data_split_method": self.data_split_method,
            "enable_global_explain": self.enable_global_explain,
            "xgboost_version": self.xgboost_version,
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
