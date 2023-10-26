# Contains code from https://github.com/pandas-dev/pandas/blob/main/pandas/core/reshape/encoding.py
"""
Encoding routines
"""
from __future__ import annotations

from bigframes import constants


def get_dummies(
    data,
    prefix=None,
    prefix_sep="_",
    dummy_na=False,
    columns=None,
    sparse=False,
    drop_first=False,
    dtype=None,
):
    """
    Convert categorical variable into dummy/indicator variables.

    Each variable is converted in as many 0/1 variables as there are
    different values. Columns in the output are each named after a value;
    if the input is a DataFrame, the name of the original variable is
    prepended to the value.

    Args:
      data (array-like, Series, or DataFrame):
        Data of which to get dummy indicators.

      prefix (str, list of str, or dict of str, default None):
        String to append DataFrame column names. Pass a list with length
        equal to the number of columns when calling get_dummies on a
        DataFrame. Alternatively, prefix can be a dictionary mapping column
        names to prefixes.

      prefix_sep (str, default '_'):
        appending prefix, separator/delimiter to use. Or pass a list or
        dictionary as with prefix.

      dummy_na (bool, default False):
        Add a column to indicate NaNs, if False NaNs are ignored.

      columns (list-like, default None):
        Column names in the DataFrame to be encoded. If columns is None
        then only the columns with string dtype will be converted.

      sparse (bool, default False):
        All BigQuery DataFrames have the same backing- sparse arg not supported.

      drop_first (bool, default False):
        Whether to get k-1 dummies out of k categorical levels by removing the
        first level.

      dtype (dtype, default bool):
        Data type for new columns. Only a single dtype is allowed.

    Returns:
      DataFrame: Dummy-coded data. If data contains other columns than the
      dummy-coded one(s), these will be prepended, unaltered, to the
      result.
    """
    raise NotImplementedError(constants.ABSTRACT_METHOD_ERROR_MESSAGE)
