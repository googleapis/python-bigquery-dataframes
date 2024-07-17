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
from pathlib import Path
import time

import bigframes_vendored.db_benchmark.groupby_queries as vendored_dbbenchmark_groupby_queries

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--table_id",
        type=str,
        required=True,
        help="The BigQuery table ID to query.",
    )
    parser.add_argument(
        "--benchmark_suffix",
        type=str,
        help="Suffix to append to benchmark names for identification purposes.",
    )
    args = parser.parse_args()

    start_time = time.perf_counter()
    vendored_dbbenchmark_groupby_queries.q7(args.table_id)
    end_time = time.perf_counter()
    runtime = end_time - start_time

    current_path = Path(__file__).absolute()
    suffix = args.benchmark_suffix

    clock_time_file_path = f"{current_path}_{suffix}.local_exec_time_seconds"

    with open(clock_time_file_path, "w") as log_file:
        log_file.write(f"{runtime}\n")
