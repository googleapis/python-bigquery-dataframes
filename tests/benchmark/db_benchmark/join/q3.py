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

import bigframes_vendored.db_benchmark.join_queries as vendored_dbbenchmark_join_queries

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--table_id",
        type=str,
        required=True,
        help="The BigQuery table ID to query.",
    )
    args = parser.parse_args()

    vendored_dbbenchmark_join_queries.q3(args.table_id)
