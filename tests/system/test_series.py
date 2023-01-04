import numpy
import pytest


@pytest.mark.parametrize(
    ["col_name", "expected_dtype"],
    [
        # TODO(swast): Use pandas.BooleanDtype() to represent nullable bool.
        ("bool_col", numpy.dtype("object")),
        # TODO(swast): Use a more efficient type.
        ("bytes_col", numpy.dtype("object")),
        # TODO(swast): Update ibis-bigquery backend to use
        # db_dtypes.DateDtype() when available.
        ("date_col", numpy.dtype("datetime64[ns]")),
        ("datetime_col", numpy.dtype("datetime64[ns]")),
        ("float64_col", numpy.dtype("float64")),
        ("geography_col", numpy.dtype("object")),
        # TODO(swast): Don't accidentally discard data if NULL is present by
        # casting to float.
        ("int64_col", numpy.dtype("float64")),
        # TODO(swast): Use a more efficient type.
        ("numeric_col", numpy.dtype("object")),
        # TODO(swast): Use a consistent dtype for integers whether NULL is
        # present or not.
        ("rowindex", numpy.dtype("int64")),
        # TODO(swast): Use a more efficient type.
        ("string_col", numpy.dtype("object")),
        # TODO(swast): Update ibis-bigquery backend to use
        # db_dtypes.TimeDtype() when available.
        ("time_col", numpy.dtype("object")),
        # TODO(swast): Make sure timestamps are associated with UTC timezone.
        ("timestamp_col", numpy.dtype("datetime64[ns]")),
    ],
)
def test_get_column(scalars_df, scalars_load_job, col_name, expected_dtype):
    series = scalars_df[col_name]
    series_pandas = series.compute()
    assert series_pandas.dtype == expected_dtype
    # TODO(swast): Compare lengths with DataFrame length computed by Bigframes.
    assert series_pandas.shape[0] == scalars_load_job.output_rows
