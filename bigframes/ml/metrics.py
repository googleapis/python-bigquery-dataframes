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

"""Implements Scikit-Learn's sklearn.metrics API"""

import typing
from typing import Tuple

import pandas as pd

import bigframes


def r2_score(
    y_true: bigframes.DataFrame, y_pred: bigframes.DataFrame, force_finite=True
) -> float:
    """Compute the R^2 (coefficient of determination) regression score.

    Perfect prediction will yield a score of 1.0, always predicting the average
    will yield 0.0, and worse prediction will get a negative score.

    If all of y_true has the same value, the score would be NaN or -Inf; to avoid
    this 1.0 or 0.0 respectively will be returned instead. This behavior can be
    disabled by setting force_finite=False"""
    # TODO(bmil): support multioutput
    if len(y_true.columns) > 1 or len(y_pred.columns) > 1:
        raise NotImplementedError(
            "Only one labels column, one predictions column is supported"
        )

    y_true_series = typing.cast(
        bigframes.Series, y_true[typing.cast(str, y_true.columns.tolist()[0])]
    )
    y_pred_series = typing.cast(
        bigframes.Series, y_pred[typing.cast(str, y_pred.columns.tolist()[0])]
    )

    # total sum of squares
    # TODO(bmil): remove .compute() when bigframes supports
    # (dataframe, scalar) binops
    # TODO(bmil): remove multiply by self when bigframes supports pow()
    delta_from_mean = y_true_series - y_true_series.mean().compute()
    ss_total = (delta_from_mean * delta_from_mean).sum().compute()

    # residual sum of squares
    # TODO(bmil): remove .compute() when bigframes supports
    # (scalar, scalar) binops
    # TODO(bmil): remove multiply by self when bigframes supports pow()
    delta_from_pred = y_true_series - y_pred_series
    ss_res = (delta_from_pred * delta_from_pred).sum().compute()

    if force_finite and ss_total == 0:
        return 0.0 if ss_res > 0 else 1.0

    return 1 - (ss_res / ss_total)


def roc_curve(
    y_true: bigframes.DataFrame,
    y_score: bigframes.DataFrame,
    drop_intermediate: bool = True,
) -> Tuple[bigframes.Series, bigframes.Series, bigframes.Series]:
    """Compute the Receiver Operating Characteristic Curve (ROC Curve) from
    prediction scores.

    The ROC Curve is the plot of true positive rate versus false positive
    rate, parameterized by the confidence threshold.

    Args:
        y_true: Binary indicators.

        y_score: Prediction scores.

            For binary predictions this may be a single column containing
            either probability estimates, or the decision value.

        drop_intermediate: Whether to exclude some intermediate thresholds to
            create a lighter curve.

    Returns:
        fpr: Increasing false positive rates such that element i is the fpr
            of predictions with score >= thresholds[i]

        tpr: Increasing true positive rates such that element i is the tpr
            of predictions with score >= thresholds[i]

        thresholds: Decreasing thresholds on the decision function used to
            compute fpr and tpr
    """
    # TODO(bmil): Add multi-class support
    # TODO(bmil): Add multi-label support
    if len(y_true.columns) > 1 or len(y_score.columns) > 1:
        raise NotImplementedError("Only binary classification is supported")

    # TODO(bmil): Implement drop_intermediate
    if drop_intermediate:
        raise NotImplementedError("drop_intermediate is not yet implemented")

    # TODO(bmil): remove this once bigframes supports the necessary operations
    session = y_true._block.expr._session
    pd_y_true = y_true.to_pandas()
    pd_y_score = y_score.to_pandas()

    # We operate on rows, so, remove the index if there is one
    # TODO(bmil): check that the indexes are equivalent before removing
    pd_y_true = pd_y_true.reset_index(drop=True)
    pd_y_score = pd_y_score.reset_index(drop=True)

    pd_df = pd.DataFrame(
        {
            "y_true": pd_y_true[pd_y_true.columns[0]],
            "y_score": pd_y_score[pd_y_score.columns[0]],
        }
    )

    total_positives = pd_df.y_true.sum()
    total_negatives = len(pd_df) - total_positives

    pd_df = pd_df.sort_values(by="y_score", ascending=False)
    pd_df["cum_tp"] = pd_df.y_true.cumsum()
    pd_df["cum_fp"] = (~pd_df.y_true.astype(bool)).cumsum()

    # produce just one data point per y_score
    pd_df = pd_df.groupby("y_score", as_index=False).last()
    pd_df = pd_df.sort_values(by="y_score", ascending=False)

    pd_df["tpr"] = pd_df.cum_tp / total_positives
    pd_df["fpr"] = pd_df.cum_fp / total_negatives
    pd_df["thresholds"] = pd_df.y_score

    # sklearn includes an extra datapoint for the origin with threshold max(y_score) + 1
    # TODO(bmil): is there a way to do this in BigFrames that doesn't violate SINGLE_QUERY
    # and isn't terribly inefficient?
    pd_origin = pd.DataFrame(
        {"tpr": [0.0], "fpr": [0.0], "thresholds": [pd_df["y_score"].max() + 1]}
    )
    pd_df = pd.concat([pd_origin, pd_df])

    df = session.read_pandas(pd_df)
    return df.fpr, df.tpr, df.thresholds


def roc_auc_score(y_true: bigframes.DataFrame, y_score: bigframes.DataFrame) -> float:
    """Compute the Receiver Operating Characteristic Area Under Curve
    (ROC AUC) from prediction scores.

    The ROC Curve is the plot of true positive rate versus false positive
    rate, parameterized by the confidence threshold. Random guessing
    should yield an area under the curve of 0.5, and the area will approach
    1.0 for perfect discrimination.

    Args:
        y_true: True binary indicators.

        y_score: Prediction scores.

           For binary predictions this may be a single column containing
           either probability estimates, or the decision value.
    """
    # TODO(bmil): Add multi-class support
    # TODO(bmil): Add multi-label support
    if len(y_true.columns) > 1 or len(y_score.columns) > 1:
        raise NotImplementedError("Only binary classification is supported")

    fpr, tpr, _ = roc_curve(y_true, y_score, drop_intermediate=False)

    # TODO(bmil): remove this once bigframes supports the necessary operations
    pd_fpr = fpr.compute()
    pd_tpr = tpr.compute()

    # Use the trapezoid rule to compute the area under the ROC curve
    width_diff = pd_fpr.diff().iloc[1:].reset_index(drop=True)
    height_avg = (pd_tpr.iloc[:-1] + pd_tpr.iloc[1:].reset_index(drop=True)) / 2
    return (width_diff * height_avg).sum()
