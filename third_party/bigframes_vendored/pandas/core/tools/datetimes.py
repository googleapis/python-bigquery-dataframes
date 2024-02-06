# Contains code from https://github.com/pandas-dev/pandas/blob/main/pandas/core/tools/datetimes.py

from typing import Literal


def to_datetime(
    arg,
    *,
    utc=False,
    format=None,
    unit=None,
):
    """
    This function converts a scalar, array-like or Series to a pandas datetime object.

    .. note::
        BigQuery only supports precision up to microseconds (us). Therefore, when working
        with timestamps that have a finer granularity than microseconds, be aware that
        the additional precision will not be represented in BigQuery.

    Args:
        arg (int, float, str, datetime, list, tuple, 1-d array, Series):
            The object to convert to a datetime. If a DataFrame is provided, the method
            expects minimally the following columns: "year", "month", "day". The column
            “year” must be specified in 4-digit format.

        utc (bool, default False):
            Control timezone-related parsing, localization and conversion. If True, the
            function always returns a timezone-aware UTC-localized timestamp or series.
            If False (default), inputs will not be coerced to UTC.

        format (str, default None):
            The strftime to parse time, e.g. "%d/%m/%Y".

        unit (str, default 'ns'):
            The unit of the arg (D,s,ms,us,ns) denote the unit, which is an integer or
            float number.

    Returns:
        Timestamp, datetime.datetime or bigframes.series.Series: Return type depends on input.
    """
