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

"""
Wraps primitives for machine learning with BQML

This library is an evolving attempt to
- implement BigFrames API for BQML
- follow as close as possible the API design of SKLearn
    https://arxiv.org/pdf/1309.0238.pdf
"""

import inspect
from typing import Dict


class BaseEstimator:
    """
    A BigFrames machine learning component following the SKLearn API design
    Ref: https://github.com/scikit-learn/scikit-learn/blob/7db5b6a98ac6ad0976a3364966e214926ca8098a/sklearn/base.py#L112

    The estimator is the fundamental abstraction for all learning components. This includes learning
    algorithms, and also some preprocessing routines.

    This base class provides shared methods for inspecting parameters, and for building a consistent
    string representation of the component. By convention, the __init__ of all descendents will be
    assumed to be the list of hyperparameters.

    All descendents of this class should implement:

        def __init__(self, hyperparameter_1=default_1, hyperparameter_2=default_2, hyperparameter3, ...):
            '''Set hyperparameters'''
            self.hyperparameter_1 = hyperparameter_1
            self.hyperparameter_2 = hyperparameter_2
            self.hyperparameter3 = hyperparameter3
            ...

        def fit(self, x_train: Union[DataFrame, Series], y_train: Union[DataFrame, Series]):
            '''Train the model to predict y_train from x_train'''
            ...

    The types of decendents of this class should be:

    1) Predictors
        These extend the interface with a .predict(self, x_test) method which predicts the target values
        according to the parameters that were calculated in .fit()

            def predict(self, x_test: Union[DataFrame, Series]) -> Union[DataFrame, Series]:
                '''Predict the target values according to the parameters that were calculated in .fit'''
                ...

    2) Transformers
        These extend the interface with .transform(self, x) and .fit_transform(x_train) methods, which
        apply data processing steps such as scaling that must be fitted to training data

            def transform(self, x: Union[DataFrame, Series]) -> Union[DataFrame, Series]:
                '''Transform the data according to the parameters that were calculated in .fit()'''
                ...

            def fit_transform(self, x_train: Union[DataFrame, Series], y_train: Union[DataFrame, Series]):
                '''Perform both fit() and transform()'''
                ...

    By design, implementations of these to inherit from are not provided, as they have no shared logic.
    If models do have shared logic, they should share it with mixin classes.
    """

    @classmethod
    def __get_param_names(cls):
        """Inspect the estimator's constructor to retrieve parameter names"""
        parameters = inspect.signature(cls.__init__).parameters.values()
        return sorted([p.name for p in parameters if p.name != "self"])

    def get_params(self) -> Dict:
        """Print the parameters for this estimator"""
        return dict([(key, getattr(self, key)) for key in self.__get_param_names()])

    def __repr__(self):
        return "I can't beleive it's not SKLearn!"
