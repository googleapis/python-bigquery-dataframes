# Contains code from https://github.com/pandas-dev/pandas/blob/main/pandas/core/reshape/tile.py
"""
Quantilization functions and related routines
"""
from __future__ import annotations

from bigframes import constants


def cut(
    x,
    bins,
    *,
    labels=None,
):
    """
    Bin values into discrete intervals.

    Use `cut` when you need to segment and sort data values into bins. This
    function is also useful for going from a continuous variable to a
    categorical variable. For example, `cut` could convert ages to groups of
    age ranges. Supports binning into an equal number of bins, or a
    pre-specified array of bins.

    ``labels=False`` implies you just want the bins back.

    **Examples:**

        >>> import bigframes.pandas as bpd
        >>> bpd.options.display.progress_bar = None
        >>> s = bpd.Series([0, 1, 5, 10])
        >>> s
        0     0
        1     1
        2     5
        3    10
        dtype: Int64

    Cut with an integer (equal-width bins):

        >>> bpd.cut(s, bins=4)
            0    {'left_exclusive': -0.01, 'right_inclusive': 2.5}
            1    {'left_exclusive': -0.01, 'right_inclusive': 2.5}
            2      {'left_exclusive': 2.5, 'right_inclusive': 5.0}
            3     {'left_exclusive': 7.5, 'right_inclusive': 10.0}
            dtype: struct<left_exclusive: double, right_inclusive: double>[pyarrow]

    Cut with an integer (equal-width bins) and labels=False:

        >>> bpd.cut(s, bins=4, labels=False)
        0    0
        1    0
        2    1
        3    3
        dtype: Int64

    Cut with pd.IntervalIndex, requires importing pandas for IntervalIndex:

        >>> import pandas as pd

        >>> interval_index = pd.IntervalIndex.from_tuples([(0, 1), (1, 5), (5, 20)])
        >>> bpd.cut(s, bins=interval_index)
        0                                            <NA>
        1     {'left_exclusive': 0, 'right_inclusive': 1}
        2     {'left_exclusive': 1, 'right_inclusive': 5}
        3    {'left_exclusive': 5, 'right_inclusive': 20}
        dtype: struct<left_exclusive: int64, right_inclusive: int64>[pyarrow]

    Cut with an iterable of tuples:

        >>> bins_tuples = [(0, 1), (1, 4), (5, 20)]
        >>> bpd.cut(s, bins=bins_tuples)
        0                                            <NA>
        1     {'left_exclusive': 0, 'right_inclusive': 1}
        2                                            <NA>
        3    {'left_exclusive': 5, 'right_inclusive': 20}
        dtype: struct<left_exclusive: int64, right_inclusive: int64>[pyarrow]

    Args:
        x (Series):
            The input Series to be binned. Must be 1-dimensional.
        bins (int, pd.IntervalIndex, Iterable[Tuple[Union[int, float], Union[int, float]]]):
            The criteria to bin by.

            int: Defines the number of equal-width bins in the range of `x`. The
            range of `x` is extended by .1% on each side to include the minimum
            and maximum values of `x`.

            pd.IntervalIndex or Iterable of tuples: Defines the exact bins to be used.
            It's important to ensure that these bins are non-overlapping.
        labels (None):
            Specifies the labels for the returned bins. Must be the same length as
            the resulting bins. If False, returns only integer indicators of the
            bins. This affects the type of the output container.

    Returns:
        Series: A Series representing the respective bin for each value
            of `x`. The type depends on the value of `labels`.
            sequence of scalars : returns a Series for Series `x` or a
            Categorical for all other inputs. The values stored within
            are whatever the type in the sequence is.
            False : returns an ndarray of integers.
    """
    raise NotImplementedError(constants.ABSTRACT_METHOD_ERROR_MESSAGE)


def qcut(x, q, *, labels=None, duplicates="error"):
    """
    Quantile-based discretization function.

    Discretize variable into equal-sized buckets based on rank or based
    on sample quantiles. For example 1000 values for 10 quantiles would
    produce a Categorical object indicating quantile membership for each data point.

    Args:
        x (Series):
            The input Series to be binned. Must be 1-dimensional.
        q (int or list-like of float):
            Number of quantiles. 10 for deciles, 4 for quartiles, etc. Alternately
            array of quantiles, e.g. [0, .25, .5, .75, 1.] for quartiles.
        labels (None):
            Used as labels for the resulting bins. Must be of the same length as
            the resulting bins. If False, return only integer indicators of the
            bins. If True, raises an error.
        duplicates ({default 'raise', 'drop'}, optional):
            If bin edges are not unique, raise ValueError or drop non-uniques.

    Returns:
        Series: Categorical or Series of integers if labels is False
            The return type (Categorical or Series) depends on the input: a Series
            of type category if input is a Series else Categorical. Bins are
            represented as categories when categorical data is returned.
    """
    raise NotImplementedError(constants.ABSTRACT_METHOD_ERROR_MESSAGE)
