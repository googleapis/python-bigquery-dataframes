# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import importlib.util
import inspect
import pathlib
import time

import bigframes


def get_configuration(include_table_id=False):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project_id",
        type=str,
        required=True,
        help="The BigQuery project ID.",
    )
    parser.add_argument(
        "--dataset_id",
        type=str,
        required=True,
        help="The BigQuery dataset ID.",
    )

    if include_table_id:
        parser.add_argument(
            "--table_id",
            type=str,
            required=True,
            help="The BigQuery table ID to query.",
        )

    parser.add_argument(
        "--ordered",
        type=str,
        help="Set to True (default) to have an ordered session, or False for an unordered session.",
    )
    parser.add_argument(
        "--benchmark_suffix",
        type=str,
        help="Suffix to append to benchmark names for identification purposes.",
    )

    args = parser.parse_args()
    session = _initialize_session(_str_to_bool(args.ordered))

    if include_table_id:
        return (
            args.project_id,
            args.dataset_id,
            args.table_id,
            session,
            args.benchmark_suffix,
        )
    else:
        return (
            args.project_id,
            args.dataset_id,
            session,
            args.benchmark_suffix,
        )


def get_execution_time(func, current_path, suffix, *args, **kwargs):
    start_time = time.perf_counter()
    func(*args, **kwargs)
    end_time = time.perf_counter()
    runtime = end_time - start_time

    clock_time_file_path = f"{current_path}_{suffix}.local_exec_time_seconds"

    with open(clock_time_file_path, "a") as log_file:
        log_file.write(f"{runtime}\n")


def import_local_module(module_name, base_path=pathlib.Path.cwd() / "third_party"):
    """
    Dynamically imports the latest benchmark scripts from a specified local directory,
    allowing these scripts to be used across different versions of libraries. This setup
    ensures that benchmark tests can be conducted using the most up-to-date scripts,
    irrespective of the library version being tested.
    """
    relative_path = pathlib.Path(*module_name.split("."))
    module_file_path = base_path / relative_path.with_suffix(".py")
    spec = importlib.util.spec_from_file_location(module_name, module_file_path)
    if spec is None:
        raise ImportError(f"Cannot load module {module_name} from {base_path}")

    module = importlib.util.module_from_spec(spec)

    assert spec.loader is not None
    spec.loader.exec_module(module)

    return module


def _str_to_bool(value):
    if value == "True":
        return True
    elif value == "False":
        return False
    else:
        raise argparse.ArgumentTypeError('Only "True" or "False" expected.')


def _initialize_session(ordered: bool):
    options_signature = inspect.signature(bigframes.BigQueryOptions.__init__)
    if "ordering_mode" in options_signature.parameters:
        context = bigframes.BigQueryOptions(
            location="US", ordering_mode="strict" if ordered else "partial"
        )
    # Older versions of bigframes
    elif "_strictly_ordered" in options_signature.parameters:
        context = bigframes.BigQueryOptions(location="US", _strictly_ordered=ordered)  # type: ignore
    elif not ordered:
        raise ValueError("Unordered mode not supported")
    else:
        context = bigframes.BigQueryOptions(location="US")
    session = bigframes.Session(context=context)
    print(f"Initialized {'ordered' if ordered else 'unordered'} session.")
    return session
