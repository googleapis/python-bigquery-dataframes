# Contains code from https://github.com/pandas-dev/pandas/blob/main/pandas/core/indexes/datetimes.py

from __future__ import annotations

from bigframes_vendored import constants
from bigframes_vendored.pandas.core.indexes import base


class DatetimeIndex(base.Index):
    """Immutable sequence used for indexing and alignment with datetime-like values"""

    @property
    def year(self) -> base.Index:
        """The year of the datetime

        **Examples:**

            >>> import bigframes.pandas as bpd
            >>> import pandas as pd
            >>> bpd.options.display.progress_bar = None

            >>> idx = bpd.Index([pd.Timestamp("20250215")])
            >>> idx.year
            Index([2025], dtype='Int64')
        """
        raise NotImplementedError(constants.ABSTRACT_METHOD_ERROR_MESSAGE)

    @property
    def month(self) -> base.Index:
        """The month as January=1, December=12.

        **Examples:**

            >>> import bigframes.pandas as bpd
            >>> import pandas as pd
            >>> bpd.options.display.progress_bar = None

            >>> idx = bpd.Index([pd.Timestamp("20250215")])
            >>> idx.month
            Index([2], dtype='Int64')
        """
        raise NotImplementedError(constants.ABSTRACT_METHOD_ERROR_MESSAGE)

    @property
    def day(self) -> base.Index:
        """The day of the datetime.

        **Examples:**

            >>> import bigframes.pandas as bpd
            >>> import pandas as pd
            >>> bpd.options.display.progress_bar = None

            >>> idx = bpd.Index([pd.Timestamp("20250215")])
            >>> idx.day
            Index([15], dtype='Int64')
        """
        raise NotImplementedError(constants.ABSTRACT_METHOD_ERROR_MESSAGE)

    @property
    def day_of_week(self) -> base.Index:
        """The day of the week with Monday=0, Sunday=6.

        **Examples:**

            >>> import bigframes.pandas as bpd
            >>> import pandas as pd
            >>> bpd.options.display.progress_bar = None

            >>> idx = bpd.Index([pd.Timestamp("20250215")])
            >>> idx.day_of_week
            Index([5], dtype='Int64')
        """
        raise NotImplementedError(constants.ABSTRACT_METHOD_ERROR_MESSAGE)

    @property
    def dayofweek(self) -> base.Index:
        """The day of the week with Monday=0, Sunday=6.

        **Examples:**

            >>> import bigframes.pandas as bpd
            >>> import pandas as pd
            >>> bpd.options.display.progress_bar = None

            >>> idx = bpd.Index([pd.Timestamp("20250215")])
            >>> idx.dayofweek
            Index([5], dtype='Int64')
        """
        raise NotImplementedError(constants.ABSTRACT_METHOD_ERROR_MESSAGE)

    @property
    def weekday(self) -> base.Index:
        """The day of the week with Monday=0, Sunday=6.

        **Examples:**

            >>> import bigframes.pandas as bpd
            >>> import pandas as pd
            >>> bpd.options.display.progress_bar = None

            >>> idx = bpd.Index([pd.Timestamp("20250215")])
            >>> idx.weekday
            Index([5], dtype='Int64')
        """
        raise NotImplementedError(constants.ABSTRACT_METHOD_ERROR_MESSAGE)


def date_range(
    start=None,
    end=None,
    periods=None,
    freq=None,
    tz=None,
    normalize: bool = False,
    name=None,
    inclusive="both",
    *,
    unit: str | None = None,
    **kwargs,
) -> DatetimeIndex:
    """
    Return a fixed frequency DatetimeIndex.

    Returns the range of equally spaced time points (where the difference between any
    two adjacent points is specified by the given frequency) such that they fall in the
    range `[start, end]` , where the first one and the last one are, resp., the first
    and last time points in that range that fall on the boundary of ``freq`` (if given
    as a frequency string) or that are valid for ``freq`` (if given as a
    :class:`pandas.tseries.offsets.DateOffset`). If ``freq`` is positive, the points
    satisfy `start <[=] x <[=] end`, and if ``freq`` is negative, the points satisfy
    `end <[=] x <[=] start`. (If exactly one of ``start``, ``end``, or ``freq`` is *not*
    specified, this missing parameter can be computed given ``periods``, the number of
    timesteps in the range.)

    Of the four parameters ``start``, ``end``, ``periods``, and ``freq``,
    exactly three must be specified. If ``freq`` is omitted, the resulting
    ``DatetimeIndex`` will have ``periods`` linearly spaced elements between
    ``start`` and ``end`` (closed on both sides).

    To learn more about the frequency strings, please see
    :ref:`this link<timeseries.offset_aliases>`.

    **Examples:**

        >>> import bigframes.pandas as bpd
        >>> import pandas as pd
        >>> bpd.options.display.progress_bar = None

    **Specifying the values**

    The next four examples generate the same `DatetimeIndex`, but vary
    the combination of `start`, `end` and `periods`.

    Specify `start` and `end`, with the default daily frequency.

        >>> bpd.date_range(start="1/1/2018", end="1/08/2018")
        DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04',
                       '2018-01-05', '2018-01-06', '2018-01-07', '2018-01-08'],
                      dtype='datetime64[ns]', freq='D')

    Specify timezone-aware `start` and `end`, with the default daily frequency.

        >>> bpd.date_range(
        ...     start=pd.to_datetime("1/1/2018").tz_localize("Europe/Berlin"),
        ...     end=pd.to_datetime("1/08/2018").tz_localize("Europe/Berlin"),
        ... )
        DatetimeIndex(['2018-01-01 00:00:00+01:00', '2018-01-02 00:00:00+01:00',
                       '2018-01-03 00:00:00+01:00', '2018-01-04 00:00:00+01:00',
                       '2018-01-05 00:00:00+01:00', '2018-01-06 00:00:00+01:00',
                       '2018-01-07 00:00:00+01:00', '2018-01-08 00:00:00+01:00'],
                      dtype='datetime64[ns, Europe/Berlin]', freq='D')

    Specify `start` and `periods`, the number of periods (days).

        >>> bpd.date_range(start="1/1/2018", periods=8)
        DatetimeIndex(['2018-01-01', '2018-01-02', '2018-01-03', '2018-01-04',
                       '2018-01-05', '2018-01-06', '2018-01-07', '2018-01-08'],
                      dtype='datetime64[ns]', freq='D')

    Specify `end` and `periods`, the number of periods (days).

        >>> bpd.date_range(end="1/1/2018", periods=8)
        DatetimeIndex(['2017-12-25', '2017-12-26', '2017-12-27', '2017-12-28',
                       '2017-12-29', '2017-12-30', '2017-12-31', '2018-01-01'],
                      dtype='datetime64[ns]', freq='D')

    Specify `start`, `end`, and `periods`; the frequency is generated
    automatically (linearly spaced).

        >>> bpd.date_range(start="2018-04-24", end="2018-04-27", periods=3)
        DatetimeIndex(['2018-04-24 00:00:00', '2018-04-25 12:00:00',
                       '2018-04-27 00:00:00'],
                      dtype='datetime64[ns]', freq=None)

    **Other Parameters**

    Changed the `freq` (frequency) to ``'ME'`` (month end frequency).

        >>> bpd.date_range(start="1/1/2018", periods=5, freq="ME")
        DatetimeIndex(['2018-01-31', '2018-02-28', '2018-03-31', '2018-04-30',
                       '2018-05-31'],
                      dtype='datetime64[ns]', freq='ME')

    Multiples are allowed

        >>> bpd.date_range(start="1/1/2018", periods=5, freq="3ME")
        DatetimeIndex(['2018-01-31', '2018-04-30', '2018-07-31', '2018-10-31',
                       '2019-01-31'],
                      dtype='datetime64[ns]', freq='3ME')

    `freq` can also be specified as an Offset object.

        >>> bpd.date_range(start="1/1/2018", periods=5, freq=pd.offsets.MonthEnd(3))
        DatetimeIndex(['2018-01-31', '2018-04-30', '2018-07-31', '2018-10-31',
                       '2019-01-31'],
                      dtype='datetime64[ns]', freq='3ME')

    Specify `tz` to set the timezone.

        >>> bpd.date_range(start="1/1/2018", periods=5, tz="Asia/Tokyo")
        DatetimeIndex(['2018-01-01 00:00:00+09:00', '2018-01-02 00:00:00+09:00',
                       '2018-01-03 00:00:00+09:00', '2018-01-04 00:00:00+09:00',
                       '2018-01-05 00:00:00+09:00'],
                      dtype='datetime64[ns, Asia/Tokyo]', freq='D')

    `inclusive` controls whether to include `start` and `end` that are on the
    boundary. The default, "both", includes boundary points on either end.

        >>> bpd.date_range(start="2017-01-01", end="2017-01-04", inclusive="both")
        DatetimeIndex(['2017-01-01', '2017-01-02', '2017-01-03', '2017-01-04'],
                      dtype='datetime64[ns]', freq='D')

    Use ``inclusive='left'`` to exclude `end` if it falls on the boundary.

        >>> bpd.date_range(start="2017-01-01", end="2017-01-04", inclusive="left")
        DatetimeIndex(['2017-01-01', '2017-01-02', '2017-01-03'],
                      dtype='datetime64[ns]', freq='D')

    Use ``inclusive='right'`` to exclude `start` if it falls on the boundary, and
    similarly ``inclusive='neither'`` will exclude both `start` and `end`.

        >>> bpd.date_range(start="2017-01-01", end="2017-01-04", inclusive="right")
        DatetimeIndex(['2017-01-02', '2017-01-03', '2017-01-04'],
                      dtype='datetime64[ns]', freq='D')

    **Specify a unit**

        >>> bpd.date_range(start="2017-01-01", periods=10, freq="100YS", unit="s")
        DatetimeIndex(['2017-01-01', '2117-01-01', '2217-01-01', '2317-01-01',
                       '2417-01-01', '2517-01-01', '2617-01-01', '2717-01-01',
                       '2817-01-01', '2917-01-01'],
                      dtype='datetime64[s]', freq='100YS-JAN')

    Arguments:
        start (str or datetime-like, optional):
            Left bound for generating dates.
        end (str or datetime-like, optional):
            Right bound for generating dates.
        periods (int, optional):
            Number of periods to generate.
        freq (str, Timedelta, datetime.timedelta, or DateOffset, default 'D'):
            Frequency strings can have multiples, e.g. '5h'. See
            :ref:`here <timeseries.offset_aliases>` for a list of
            frequency aliases.
        tz (str or tzinfo, optional):
            Time zone name for returning localized DatetimeIndex. By default,
            the resulting DatetimeIndex is timezone-naive unless timezone-aware
            datetime-likes are passed.

            "UTC" is the only currently-supported timezone.
        normalize (bool, default False):
            Normalize start/end dates to midnight before generating date range.
        name (str, default None):
            Name of the resulting DatetimeIndex.
        inclusive ({"both", "neither", "left", "right"}, default "both"):
            Include boundaries; Whether to set each bound as closed or open.
        unit (str, default None):
            Specify the desired resolution of the result.

            "us" is the only currently-supported resolution.

    Returns:
        DatetimeIndex
            A DatetimeIndex object of the generated dates.
    """

    raise NotImplementedError(constants.ABSTRACT_METHOD_ERROR_MESSAGE)
