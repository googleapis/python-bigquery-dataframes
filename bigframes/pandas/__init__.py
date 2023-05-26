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

"""BigFrames provides a DataFrame API for BigQuery."""

from __future__ import annotations

import threading
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    MutableSequence,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy
import pandas

import bigframes._config as config
import bigframes.bigframes
import bigframes.dataframe
import bigframes.session

concat = bigframes.bigframes.concat
options = config.options

_global_session: Optional[bigframes.session.Session] = None
_global_session_lock = threading.Lock()


def reset_session() -> None:
    """Start a fresh session next time a function requires a session.

    Closes the current session if it was already started.
    """
    global _global_session

    with _global_session_lock:
        if _global_session is not None:
            _global_session.close()
            _global_session = None

        options.bigquery._session_started = False


# Note: the following methods are duplicated from Session. This duplication
# enables the following:
#
# 1. Static type checking knows the argument and return types, which is
#    difficult to do with decorators. Aside: When we require Python 3.10, we
#    can use Concatenate for generic typing in decorators. See:
#    https://stackoverflow.com/a/68290080/101923
# 2. docstrings get processed by static processing tools, such as VS Code's
#    autocomplete.
# 3. Positional arguments function as expected. If we were to pull in the
#    methods directly from Session, a Session object would need to be the first
#    argument, even if we allow a default value.


def _with_default_session(func, *args, **kwargs):
    global _global_session, _global_session_lock

    with _global_session_lock:
        if _global_session is None:
            _global_session = bigframes.session.connect(options.bigquery)

    return func(_global_session, *args, **kwargs)


def read_csv(
    filepath_or_buffer: str,
    *,
    sep: Optional[str] = ",",
    header: Optional[int] = 0,
    names: Optional[
        Union[MutableSequence[Any], numpy.ndarray[Any, Any], Tuple[Any, ...], range]
    ] = None,
    index_col: Optional[
        Union[int, str, Sequence[Union[str, int]], Literal[False]]
    ] = None,
    usecols: Optional[
        Union[
            MutableSequence[str],
            Tuple[str, ...],
            Sequence[int],
            pandas.Series,
            pandas.Index,
            numpy.ndarray[Any, Any],
            Callable[[Any], bool],
        ]
    ] = None,
    dtype: Optional[Dict] = None,
    engine: Optional[
        Literal["c", "python", "pyarrow", "python-fwf", "bigquery"]
    ] = None,
    encoding: Optional[str] = None,
    **kwargs,
) -> bigframes.dataframe.DataFrame:
    r"""Loads DataFrame from comma-separated values (csv) file locally or from GCS.

    The CSV file data will be persisted as a temporary BigQuery table, which can be
    automatically recycled after the Session is closed.

    Note: using `engine="bigquery"` will not guarantee the same ordering as the
    file in the resulting dataframe.

    Args:
        filepath_or_buffer: a string path including GCS and local file.

        sep: the separator for fields in a CSV file. For the BigQuery engine, the separator
            can be any ISO-8859-1 single-byte character. To use a character in the range
            128-255, you must encode the character as UTF-8. Both engines support
            `sep="\t"` to specify tab character as separator. Default engine supports
            having any number of spaces as separator by specifying `sep="\s+"`. Separators
            longer than 1 character are interpreted as regular expressions by the default
            engine. BigQuery engine only supports single character separators.

        header: row number to use as the column names.
            - ``None``: Instructs autodetect that there are no headers and data should be
            read starting from the first row.
            - ``0``: If using `engine="bigquery"`, Autodetect tries to detect headers in the
            first row. If they are not detected, the row is read as data. Otherwise data
            is read starting from the second row. When using default engine, pandas assumes
            the first row contains column names unless the `names` argument is specified.
            If `names` is provided, then the first row is ignored, second row is read as
            data, and column names are inferred from `names`.
            - ``N > 0``: If using `engine="bigquery"`, Autodetect skips N rows and tries
            to detect headers in row N+1. If headers are not detected, row N+1 is just
            skipped. Otherwise row N+1 is used to extract column names for the detected
            schema. When using default engine, pandas will skip N rows and assumes row N+1
            contains column names unless the `names` argument is specified. If `names` is
            provided, row N+1 will be ignored, row N+2 will be read as data, and column
            names are inferred from `names`.

        names: a list of column names to use. If the file contains a header row and you
            want to pass this parameter, then `header=0` should be passed as well so the
            first (header) row is ignored. Only to be used with default engine.

        index_col: column(s) to use as the row labels of the DataFrame, either given as
            string name or column index. `index_col=False` can be used with the default
            engine only to enforce that the first column is not used as the index. Using
            column index instead of column name is only supported with the default engine.
            The BigQuery engine only supports having a single column name as the `index_col`.
            Neither engine supports having a multi-column index.

        usecols: list of column names to use. The BigQuery engine only supports having a list
            of string column names. Column indices and callable functions are only supported
            with the default engine. Using the default engine, the column names in `usecols`
            can be defined to correspond to column names provided with the `names` parameter
            (ignoring the document's header row of column names). The order of the column
            indices/names in `usecols` is ignored with the default engine. The order of the
            column names provided with the BigQuery engine will be consistent in the resulting
            dataframe. If using a callable function with the default engine, only column names
            that evaluate to True by the callable function will be in the resulting dataframe.

        dtype: data type for data or columns. Only to be used with default engine.

        engine: type of engine to use. If `engine="bigquery"` is specified, then BigQuery's
            load API will be used. Otherwise, the engine will be passed to `pandas.read_csv`.

        encoding: the character encoding of the data. The default encoding is `UTF-8` for both
            engines. The default engine acceps a wide range of encodings. Refer to Python
            documentation for a comprehensive list,
            https://docs.python.org/3/library/codecs.html#standard-encodings
            The BigQuery engine only supports `UTF-8` and `ISO-8859-1`.

        **kwargs: keyword arguments.


    Returns:
        A BigFrame DataFrame.
    """
    # NOTE: Please keep this docstring in sync with the one in bigframes.session.
    return _with_default_session(
        bigframes.session.Session.read_csv,
        filepath_or_buffer=filepath_or_buffer,
        sep=sep,
        header=header,
        names=names,
        index_col=index_col,
        usecols=usecols,
        dtype=dtype,
        engine=engine,
        encoding=encoding,
        **kwargs,
    )


def read_gbq(
    query_or_table: str,
    *,
    index_cols: Iterable[str] = (),
    col_order: Iterable[str] = (),
    max_results: Optional[int] = None,
) -> bigframes.dataframe.DataFrame:
    """
    Loads DataFrame from Google BigQuery.

    Args:
        query_or_table: a SQL string to be executed or a BigQuery table to be read. The
          table must be specified in the format of `project.dataset.tablename` or
          `dataset.tablename`.
        index_cols: List of column names to use as the index or multi-index.
        col_order: List of BigQuery column names in the desired order for results DataFrame.
        max_results: Limit the maximum number of rows to fetch from the query results.

    Returns:
        A DataFrame representing results of the query or table.
    """
    # NOTE: Please keep this docstring in sync with the one in bigframes.session.
    return _with_default_session(
        bigframes.session.Session.read_gbq,
        query_or_table,
        index_cols=index_cols,
        col_order=col_order,
        max_results=max_results,
    )


def read_gbq_model(model_name: str):
    """Loads a BQML model from Google BigQuery.

    Args:
        model_name : the model's name in BigQuery in the format
        `project_id.dataset_id.model_id`, or just `dataset_id.model_id`
        to load from the default project.

    Returns:
        A bigframes.ml Model wrapping the model
    """
    # NOTE: Please keep this docstring in sync with the one in bigframes.session.
    return _with_default_session(
        bigframes.session.Session.read_gbq_model,
        model_name,
    )


def read_pandas(pandas_dataframe: pandas.DataFrame) -> bigframes.dataframe.DataFrame:
    """Loads DataFrame from a Pandas DataFrame.

    The Pandas DataFrame will be persisted as a temporary BigQuery table, which can be
    automatically recycled after the Session is closed.

    Args:
        pandas_dataframe: a Pandas DataFrame object to be loaded.

    Returns:
        A BigFrame DataFrame.
    """
    # NOTE: Please keep this docstring in sync with the one in bigframes.session.
    return _with_default_session(
        bigframes.session.Session.read_pandas,
        pandas_dataframe,
    )


def remote_function(
    input_types: List[type],
    output_type: type,
    dataset: Optional[str] = None,
    bigquery_connection: Optional[str] = None,
    reuse: bool = True,
):
    """Create a remote function from a user defined function."""
    return _with_default_session(
        bigframes.session.Session.remote_function,
        input_types=input_types,
        output_type=output_type,
        dataset=dataset,
        bigquery_connection=bigquery_connection,
        reuse=reuse,
    )


# Use __all__ to let type checkers know what is part of the public API.
__all___ = [
    "concat",
    "options",
    "read_csv",
    "read_gbq",
    "read_gbq_model",
    "read_pandas",
    "remote_function",
]
