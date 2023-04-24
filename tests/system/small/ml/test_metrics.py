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

import pandas as pd

import bigframes.ml.metrics


def test_roc_curve_binary_classification_prediction_returns_expected(session):
    pd_df = pd.DataFrame(
        {
            "y_true": [0, 0, 1, 1, 0, 1, 0, 1, 1, 1],
            "y_score": [0.1, 0.4, 0.35, 0.8, 0.65, 0.9, 0.5, 0.3, 0.6, 0.45],
        }
    )

    df = session.read_pandas(pd_df)
    fpr, tpr, thresholds = bigframes.ml.metrics.roc_curve(
        df[["y_true"]], df[["y_score"]], drop_intermediate=False
    )

    pd_fpr = fpr.compute()
    pd_tpr = tpr.compute()
    pd_thresholds = thresholds.compute()

    pd.testing.assert_series_equal(
        pd_thresholds,
        pd.Series(
            [1.9, 0.9, 0.8, 0.65, 0.6, 0.5, 0.45, 0.4, 0.35, 0.3, 0.1],
            dtype="Float64",
            name="thresholds",
        ),
        check_index_type=False,
    )
    pd.testing.assert_series_equal(
        pd_fpr,
        pd.Series(
            [0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.5, 0.75, 0.75, 0.75, 1.0],
            dtype="Float64",
            name="fpr",
        ),
        check_index_type=False,
    )
    pd.testing.assert_series_equal(
        pd_tpr,
        pd.Series(
            [
                0.0,
                0.16666667,
                0.33333333,
                0.33333333,
                0.5,
                0.5,
                0.66666667,
                0.66666667,
                0.83333333,
                1.0,
                1.0,
            ],
            dtype="Float64",
            name="tpr",
        ),
        check_index_type=False,
    )


def test_roc_curve_binary_classification_decision_returns_expected(session):
    # Instead of operating on probabilities, assume a 70% decision threshold
    # has been applied, and operate on the final output
    y_score = [0.1, 0.4, 0.35, 0.8, 0.65, 0.9, 0.5, 0.3, 0.6, 0.45]
    decisions_70pct = [1 if s > 0.7 else 0 for s in y_score]
    pd_df = pd.DataFrame(
        {
            "y_true": [0, 0, 1, 1, 0, 1, 0, 1, 1, 1],
            "y_score": decisions_70pct,
        }
    )

    df = session.read_pandas(pd_df)
    fpr, tpr, thresholds = bigframes.ml.metrics.roc_curve(
        df[["y_true"]], df[["y_score"]], drop_intermediate=False
    )

    pd_fpr = fpr.compute()
    pd_tpr = tpr.compute()
    pd_thresholds = thresholds.compute()

    pd.testing.assert_series_equal(
        pd_thresholds,
        pd.Series(
            [2.0, 1.0, 0.0],
            dtype="Int64",
            name="thresholds",
        ),
        check_index_type=False,
    )
    pd.testing.assert_series_equal(
        pd_fpr,
        pd.Series(
            [0.0, 0.0, 1.0],
            dtype="Float64",
            name="fpr",
        ),
        check_index_type=False,
    )
    pd.testing.assert_series_equal(
        pd_tpr,
        pd.Series(
            [
                0.0,
                0.33333333,
                1.0,
            ],
            dtype="Float64",
            name="tpr",
        ),
        check_index_type=False,
    )


def test_roc_auc_score_returns_expected(session):
    pd_df = pd.DataFrame(
        {
            "y_true": [0, 0, 1, 1, 0, 1, 0, 1, 1, 1],
            "y_score": [0.1, 0.4, 0.35, 0.8, 0.65, 0.9, 0.5, 0.3, 0.6, 0.45],
        }
    )

    df = session.read_pandas(pd_df)
    score = bigframes.ml.metrics.roc_auc_score(df[["y_true"]], df[["y_score"]])

    assert score == 0.625
