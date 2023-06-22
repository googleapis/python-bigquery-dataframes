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

import inspect
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


def get_global_session():
    """Gets the global session.

    Creates the global session if it does not exist.
    """
    global _global_session, _global_session_lock

    with _global_session_lock:
        if _global_session is None:
            _global_session = bigframes.session.connect(options.bigquery)

    return _global_session


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
    return func(get_global_session(), *args, **kwargs)


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


read_csv.__doc__ = inspect.getdoc(bigframes.session.Session.read_csv)


def read_gbq(
    query: str,
    *,
    index_col: Iterable[str] | str = (),
    col_order: Iterable[str] = (),
    max_results: Optional[int] = None,
) -> bigframes.dataframe.DataFrame:
    # NOTE: Please keep this docstring in sync with the one in bigframes.session.
    return _with_default_session(
        bigframes.session.Session.read_gbq,
        query,
        index_col=index_col,
        col_order=col_order,
        max_results=max_results,
    )


read_gbq.__doc__ = inspect.getdoc(bigframes.session.Session.read_gbq)


def read_gbq_model(model_name: str):
    # NOTE: Please keep this docstring in sync with the one in bigframes.session.
    return _with_default_session(
        bigframes.session.Session.read_gbq_model,
        model_name,
    )


read_gbq_model.__doc__ = inspect.getdoc(bigframes.session.Session.read_gbq_model)


def read_pandas(pandas_dataframe: pandas.DataFrame) -> bigframes.dataframe.DataFrame:
    # NOTE: Please keep this docstring in sync with the one in bigframes.session.
    return _with_default_session(
        bigframes.session.Session.read_pandas,
        pandas_dataframe,
    )


read_pandas.__doc__ = inspect.getdoc(bigframes.session.Session.read_pandas)


def remote_function(
    input_types: List[type],
    output_type: type,
    dataset: Optional[str] = None,
    bigquery_connection: Optional[str] = None,
    reuse: bool = True,
):
    return _with_default_session(
        bigframes.session.Session.remote_function,
        input_types=input_types,
        output_type=output_type,
        dataset=dataset,
        bigquery_connection=bigquery_connection,
        reuse=reuse,
    )


remote_function.__doc__ = inspect.getdoc(bigframes.session.Session.remote_function)


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
