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

import re
from unittest import mock

import pandas as pd
import pytest

from bigframes.ml import forecasting
import bigframes.pandas as bpd


@pytest.fixture(scope="module")
def polars_session():
    from .. import polars_session

    session = polars_session.TestSession()
    session._temp_storage_manager = mock.Mock()
    return session


@pytest.fixture
def mock_y(polars_session):
    pd_df_y = pd.DataFrame({"num_trips": [1, None, 3]})
    bf_df_y = bpd.DataFrame(pd_df_y, session=polars_session)
    return bf_df_y


@pytest.fixture
def mock_X(polars_session):
    pd_df_x = pd.DataFrame(
        {
            "date": pd.to_datetime(
                [
                    "2017-07-30 00:00:00+00:00",
                    "2017-03-04 00:00:00+00:00",
                    "2017-05-12 00:00:00+00:00",
                ]
            ),
            "station_id": ["132", "836", "109"],
        }
    )
    bf_df_x = bpd.DataFrame(pd_df_x, session=polars_session)
    return bf_df_x


def test_predict_explain_low_confidence_level():
    confidence_level = -0.5

    model = forecasting.ARIMAPlus()

    with pytest.raises(
        ValueError,
        match=re.escape(
            f"confidence_level must be [0.0, 1.0), but is {confidence_level}."
        ),
    ):
        model.predict_explain(horizon=4, confidence_level=confidence_level)


def test_predict_high_explain_confidence_level():
    confidence_level = 2.1

    model = forecasting.ARIMAPlus()

    with pytest.raises(
        ValueError,
        match=re.escape(
            f"confidence_level must be [0.0, 1.0), but is {confidence_level}."
        ),
    ):
        model.predict_explain(horizon=4, confidence_level=confidence_level)


def test_predict_explain_low_horizon():
    horizon = -1

    model = forecasting.ARIMAPlus()

    with pytest.raises(
        ValueError, match=f"horizon must be at least 1, but is {horizon}."
    ):
        model.predict_explain(horizon=horizon, confidence_level=0.9)


def test_forecast_fit(bqml_model_factory, mock_session, mock_X, mock_y):
    model = forecasting.ARIMAPlus()
    model._bqml_model_factory = bqml_model_factory
    model.fit(mock_X, mock_y)

    mock_session._start_query_ml_ddl.assert_called_once_with(
        "CREATE OR REPLACE MODEL `test-project`.`_anon123`.`temp_model_id`\nOPTIONS(\n  model_type='LINEAR_REG',\n  data_split_method='NO_SPLIT',\n  optimize_strategy='auto_strategy',\n  fit_intercept=True,\n  l2_reg=0.0,\n  max_iterations=20,\n  learn_rate_strategy='line_search',\n  min_rel_progress=0.01,\n  calculate_p_values=False,\n  enable_global_explain=False,\n  INPUT_LABEL_COLS=['input_column_label'])\nAS input_X_y_no_index_sql"
    )
