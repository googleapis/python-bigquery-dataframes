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

"""Metrics functions for evaluating models. This module is styled after
Scikit-Learn's metrics module: https://scikit-learn.org/stable/modules/metrics.html"""

import typing
from typing import Literal, Tuple

import pandas as pd
import sklearn.metrics as sklearn_metrics  # type: ignore

import bigframes
import bigframes.core.blocks as blocks


def r2_score(
    y_true: bigframes.DataFrame, y_pred: bigframes.DataFrame, force_finite=True
) -> float:
    """Compute the R^2 (coefficient of determination) regression score.

    Perfect prediction will yield a score of 1.0, always predicting the average
    will yield 0.0, and worse prediction will get a negative score.

    If all of y_true has the same value, the score would be NaN or -Inf; to avoid
    this 1.0 or 0.0 respectively will be returned instead. This behavior can be
    disabled by setting force_finite=False

    Args:
        y_true: Ground truth (correct) target values.

        y_score: Estimated target values.

        force_finite: Flag indicating if NaN and -Inf scores resulting from constant
            data should be replaced with real numbers (1.0 if prediction is perfect,
            0.0 otherwise). Default is True.

    Returns: the R^2 score."""
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
    # (dataframe, scalar) binops
    # TODO(bmil): remove multiply by self when bigframes supports pow()
    # TODO(tbergeron): These stats are eagerly evaluated. Move to lazy representation once scalar subqueries supported.
    delta_from_mean = y_true_series - y_true_series.mean()
    ss_total = (delta_from_mean * delta_from_mean).sum()

    # residual sum of squares
    # (scalar, scalar) binops
    # TODO(bmil): remove multiply by self when bigframes supports pow()
    delta_from_pred = y_true_series - y_pred_series
    ss_res = (delta_from_pred * delta_from_pred).sum()

    if force_finite and ss_total == 0:
        return 0.0 if ss_res > 0 else 1.0

    return 1 - (ss_res / ss_total)


def accuracy_score(
    y_true: bigframes.DataFrame, y_pred: bigframes.DataFrame, normalize=True
) -> float:
    """Compute the accuracy classification score.

    It calculates subset accuracy, where the predicted set of labels for a
    sample must precisely match the corresponding set of labels in y_true.
    This is the fraction of correct predictions. If normalize=False, it will
    instead return the total number of correct predictions.

    Args:
        y_true: Ground truth (correct) labels.

        y_score: Predicted labels, as returned by a classifier.

        normalize: If False, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.

    Returns: If normalize == True, return the fraction of correctly classified
            samples, else returns the number of correctly classified samples.

    """
    # TODO(ashleyxu): support sample_weight as the parameter
    if len(y_true.columns) != 1 or len(y_pred.columns) != 1:
        raise NotImplementedError(
            "Only one labels column, one predictions column is supported"
        )

    y_true_series = typing.cast(
        bigframes.Series, y_true[typing.cast(str, y_true.columns.tolist()[0])]
    )
    y_pred_series = typing.cast(
        bigframes.Series, y_pred[typing.cast(str, y_pred.columns.tolist()[0])]
    )

    # Compute accuracy for each possible representation
    # TODO(ashleyxu): add multilabel classification support where y_type
    # starts with "multilabel"
    score = (y_true_series == y_pred_series).astype(pd.Int64Dtype())

    if normalize:
        return score.mean()
    else:
        return score.sum()


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

    Returns: Area Under the Curve score.
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


def auc(
    x: bigframes.DataFrame,
    y: bigframes.DataFrame,
) -> float:
    """
    Compute Area Under the Curve (AUC) using the trapezoidal rule.

    Args:
        x: X coordinates. These must be either monotonic increasing or monotonic decreasing.

        y: Y coordinates.

    Returns: Area Under the Curve.
    """
    if len(x.columns) != 1 or len(y.columns) != 1:
        raise ValueError("Only 1-D data structure is supported")

    # TODO(b/286410053) Support ML exceptions and error handling.
    auc = sklearn_metrics.auc(x.to_pandas(), y.to_pandas())
    return auc


def confusion_matrix(
    y_true: bigframes.DataFrame,
    y_pred: bigframes.DataFrame,
) -> pd.DataFrame:
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true: Ground truth target values.

        y_pred: Estimated targets as returned by a classifier.

    Returns: Confusion matrix.
    """
    # TODO(ashleyxu): support labels and sample_weight parameters
    # TODO(ashleyxu): support bigframes.Series as input type
    if len(y_true.columns) != 1 or len(y_pred.columns) != 1:
        raise NotImplementedError(
            "Only one labels column, one predictions column is supported"
        )

    y_true_column = typing.cast(blocks.Label, y_true.columns[0])
    y_pred_series = typing.cast(
        bigframes.Series, y_pred[typing.cast(blocks.Label, y_pred.columns.tolist()[0])]
    )
    confusion_df = y_true.assign(y_pred=y_pred_series)
    confusion_df = confusion_df.assign(dummy=0)
    groupby_count = (
        confusion_df.groupby(by=[y_true_column, "y_pred"], as_index=False)
        .count()
        .to_pandas()
    )

    unique_values = sorted(set(groupby_count.y_true).union(set(groupby_count.y_pred)))

    confusion_matrix = pd.DataFrame(
        0, index=pd.Index(unique_values), columns=pd.Index(unique_values), dtype=int
    )

    # Loop through the result by rows and columns
    for _, row in groupby_count.iterrows():
        y_true = row["y_true"]
        y_pred = row["y_pred"]
        count = row["dummy"]
        confusion_matrix[y_pred][y_true] = count

    return confusion_matrix


def recall_score(
    y_true: bigframes.DataFrame,
    y_pred: bigframes.DataFrame,
    average: Literal["micro", "macro", "binary", "weighted", "sample", None] = "binary",
) -> pd.Series:
    """Compute recall score.

    Args:
        y_true: Ground truth target values.

        y_pred: Estimated targets as returned by a classifier.

        average: Type of averaging performed on the data. Possible values:"micro",
        "macro", "binary", "weighted", "sample", None. Default to “binary”.

    Returns: recall score.
    """
    # TODO(ashleyxu): support more average type, default to "binary"
    # TODO(ashleyxu): support bigframes.Series as input type
    if len(y_true.columns) != 1 or len(y_pred.columns) != 1:
        raise NotImplementedError(
            "Only one labels column, one predictions column is supported"
        )

    if average is not None:
        raise NotImplementedError("Only average=None is supported")

    y_true_series = typing.cast(
        bigframes.Series, y_true[typing.cast(blocks.Label, y_true.columns.tolist()[0])]
    )
    y_pred_series = typing.cast(
        bigframes.Series, y_pred[typing.cast(blocks.Label, y_pred.columns.tolist()[0])]
    )

    is_accurate = y_true_series == y_pred_series
    unique_labels = (
        bigframes.concat([y_true_series, y_pred_series], join="outer")
        .drop_duplicates()
        .sort_values()
    )
    index = unique_labels.to_list()

    recall = (
        is_accurate.groupby(y_true_series).sum()
        / is_accurate.groupby(y_true_series).count()
    ).compute()

    recall_score = pd.Series(0, index=index)
    for i in recall_score.index:
        recall_score.loc[i] = recall.loc[i]

    return recall_score


def precision_score(
    y_true: bigframes.DataFrame,
    y_pred: bigframes.DataFrame,
    average: Literal["micro", "macro", "binary", "weighted", "sample", None] = "binary",
) -> pd.Series:
    """Compute precision score.

    Args:
        y_true: Ground truth target values.

        y_pred: Estimated targets as returned by a classifier.

        average: Type of averaging performed on the data. Possible values:"micro",
        "macro", "binary", "weighted", "sample", None. Default to "binary".

    Returns: precision score.
    """
    # TODO(ashleyxu): support more average type, default to "binary"
    # TODO(ashleyxu): support bigframes.Series as input type
    if len(y_true.columns) != 1 or len(y_pred.columns) != 1:
        raise NotImplementedError(
            "Only one labels column, one predictions column is supported"
        )

    if average is not None:
        raise NotImplementedError("Only average=None is supported")

    y_true_series = typing.cast(
        bigframes.Series, y_true[typing.cast(blocks.Label, y_true.columns.tolist()[0])]
    )
    y_pred_series = typing.cast(
        bigframes.Series, y_pred[typing.cast(blocks.Label, y_pred.columns.tolist()[0])]
    )

    is_accurate = y_true_series == y_pred_series
    unique_labels = (
        bigframes.concat([y_true_series, y_pred_series], join="outer")
        .drop_duplicates()
        .sort_values()
    )
    index = unique_labels.to_list()

    precision = (
        is_accurate.groupby(y_pred_series).sum()
        / is_accurate.groupby(y_pred_series).count()
    ).compute()

    precision_score = pd.Series(0, index=index)
    for i in precision.index:
        precision_score.loc[i] = precision.loc[i]

    return precision_score


def f1_score(
    y_true: bigframes.DataFrame,
    y_pred: bigframes.DataFrame,
    average: Literal["micro", "macro", "binary", "weighted", "sample", None] = "binary",
) -> pd.Series:
    """Compute f1 score.

    Args:
        y_true: Ground truth target values.

        y_pred: Estimated targets as returned by a classifier.

        average: Type of averaging performed on the data. Possible values:"micro",
        "macro", "binary", "weighted", "sample", None. Default to "binary".

    Returns: f1 score.
    """
    # TODO(ashleyxu): support more average type, default to "binary"
    # TODO(ashleyxu): support bigframes.Series as input type
    if len(y_true.columns) != 1 or len(y_pred.columns) != 1:
        raise NotImplementedError(
            "Only one labels column, one predictions column is supported"
        )

    if average is not None:
        raise NotImplementedError("Only average=None is supported")

    recall = recall_score(y_true, y_pred, average=None)
    precision = precision_score(y_true, y_pred, average=None)

    f1_score = pd.Series(0, index=recall.index)
    for index in recall.index:
        if precision[index] + recall[index] != 0:
            f1_score[index] = (
                2
                * (precision[index] * recall[index])
                / (precision[index] + recall[index])
            )
        else:
            f1_score[index] = 0

    return f1_score
