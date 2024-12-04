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
# from unittest import mock

# from google.cloud import bigquery
import pytest

from bigframes.ml import forecasting

# import bigframes.pandas as bpd
# from . import resources


def test_predict_explain_confidence_level():
    confidence_level = 0.9

    forecasting.ARIMAPlus.predict_explain(horizon=4, confidence_level=confidence_level)


def test_predict_explain_low_confidence_level():
    confidence_level = -0.5

    with pytest.raises(
        ValueError,
        match=f"confidence_level must be 0.0 and 1.0, but is {confidence_level}.",
    ):
        forecasting.ARIMAPlus.predict_explain(
            horizon=4, confidence_level=confidence_level
        )


def test_predict_high_explain_confidence_level():
    confidence_level = 2.1

    with pytest.raises(
        ValueError,
        match=f"confidence_level must be 0.0 and 1.0, but is {confidence_level}.",
    ):
        forecasting.ARIMAPlus.predict_explain(
            horizon=4, confidence_level=confidence_level
        )


def test_predict_explain_horizon():
    horizon = 1

    forecasting.ARIMAPlus.predict_explain(horizon=horizon, confidence_level=0.9)


def test_predict_explain_low_horizon():
    horizon = 0.5

    with pytest.raises(
        ValueError, match=f"horizon must be at least 1, but is {horizon}."
    ):
        forecasting.ARIMAPlus.predict_explain(horizon=horizon, confidence_level=0.9)
