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

import functools
import inspect
import threading
from typing import List

import bigframes_vendored.constants as constants
from google.cloud import bigquery
import pandas

_lock = threading.Lock()

# The limit is 64 (https://cloud.google.com/bigquery/docs/labels-intro#requirements),
# but leave a few spare for internal labels to be added.
# See internal issue 386825477.
MAX_LABELS_COUNT = 64 - 8

_api_methods: List = []
_excluded_methods = ["__setattr__", "__getattr__"]

# Stack to track method calls
_call_stack: List = []


class UnimplementedMethodLogger:
    def __init__(self, bq_client: bigquery.Client, class_name: str, method_name: str):
        self.bq_client = bq_client
        self.class_name = class_name
        self.method_name = method_name

    def __call__(self, *args, **kwargs):
        submit_pandas_labels(
            self.bq_client,
            self.class_name,
            self.method_name,
            args,
            kwargs,
            task="pandas_api_tracking",
        )
        raise AttributeError(
            "BigQuery DataFrames has not yet implemented an equivalent to"
            f"'pandas.{self.class_name}.{self.method_name}'.\n{constants.FEEDBACK_LINK}"
        )


def submit_pandas_labels(
    bq_client: bigquery.Client,
    class_name: str,
    method_name: str,
    args,
    kwargs,
    task: str,
):
    labels_dict = {
        "task": task,
        "classname": class_name.lower(),
        "method_name": method_name.lower(),
        "args_count": len(args),
    }
    cls = getattr(pandas, class_name)
    method = getattr(cls, method_name)
    signature = inspect.signature(method)
    param_names = [param.name for param in signature.parameters.values()]

    for i, key in enumerate(kwargs.keys()):
        if len(labels_dict) >= MAX_LABELS_COUNT:
            break
        if key in param_names:
            labels_dict[f"kwargs_{i}"] = key.lower()

    if (
        len(labels_dict) == 4
        and labels_dict["args_count"] == 0
        and task == "pandas_param_tracking"
    ):
        return

    # Run a query with syntax error to avoid cost.
    query = "SELECT COUNT(x FROM data_table—"
    job_config = bigquery.QueryJobConfig(labels=labels_dict)
    bq_client.query(query, job_config=job_config)


def class_logger(decorated_cls):
    """Decorator that adds logging functionality to each method of the class."""
    for attr_name, attr_value in decorated_cls.__dict__.items():
        if callable(attr_value) and (attr_name not in _excluded_methods):
            setattr(decorated_cls, attr_name, method_logger(attr_value, decorated_cls))
        elif isinstance(attr_value, property):
            setattr(
                decorated_cls, attr_name, property_logger(attr_value, decorated_cls)
            )
    return decorated_cls


def method_logger(method, decorated_cls):
    """Decorator that adds logging functionality to a method."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        class_name = decorated_cls.__name__  # Access decorated class name
        api_method_name = str(method.__name__)
        full_method_name = f"{class_name.lower()}-{api_method_name}"

        # Track directly called methods
        if len(_call_stack) == 0:
            add_api_method(full_method_name)

        _call_stack.append(full_method_name)

        try:
            return method(self, *args, **kwargs)
        except (NotImplementedError, TypeError) as e:
            submit_pandas_labels(
                self._block.expr.session.bqclient,
                class_name,
                api_method_name,
                args,
                kwargs,
                task="pandas_param_tracking",
            )
            raise e
        finally:
            _call_stack.pop()

    return wrapper


def property_logger(prop, decorated_cls):
    """Decorator that adds logging functionality to a property."""

    def shared_wrapper(f):
        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            class_name = decorated_cls.__name__
            property_name = f.__name__
            full_property_name = f"{class_name.lower()}-{property_name.lower()}"

            if len(_call_stack) == 0:
                add_api_method(full_property_name)

            _call_stack.append(full_property_name)
            try:
                return f(*args, **kwargs)
            finally:
                _call_stack.pop()

        return wrapped

    # Apply the wrapper to the getter, setter, and deleter
    return property(
        shared_wrapper(prop.fget),
        shared_wrapper(prop.fset) if prop.fset else None,
        shared_wrapper(prop.fdel) if prop.fdel else None,
    )


def add_api_method(api_method_name):
    global _lock
    global _api_methods
    with _lock:
        # Push the method to the front of the _api_methods list
        _api_methods.insert(0, api_method_name)
        # Keep the list length within the maximum limit (adjust MAX_LABELS_COUNT as needed)
        _api_methods = _api_methods[:MAX_LABELS_COUNT]


def get_and_reset_api_methods(dry_run: bool = False):
    global _lock
    with _lock:
        previous_api_methods = list(_api_methods)

        # dry_run might not make a job resource, so only reset the log on real queries.
        if not dry_run:
            _api_methods.clear()
    return previous_api_methods
