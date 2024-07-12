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

import bigframes


def get_tpch_configuration():
    parser = argparse.ArgumentParser(description="Process TPC-H Query using BigFrames.")
    parser.add_argument(
        "--dataset_id",
        type=str,
        required=True,
        help="The BigQuery dataset ID to query.",
    )
    parser.add_argument(
        "--ordered",
        type=str,
        default=True,
        help="Set to True (default) to have an ordered session, or False for an unordered session.",
    )

    args = parser.parse_args()
    session = _initialize_session(_str_to_bool(args.ordered))
    return args.dataset_id, session


def _str_to_bool(value):
    if value == "True":
        return True
    elif value == "False":
        return False
    else:
        raise argparse.ArgumentTypeError('Only "True" or "False" expected.')


def _initialize_session(ordered: bool):
    context = bigframes.BigQueryOptions(location="US", _strictly_ordered=ordered)
    session = bigframes.Session(context=context)
    print(f"Initialized {'ordered' if ordered else 'unordered'} session.")
    return session
