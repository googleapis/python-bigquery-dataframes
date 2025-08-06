# Copyright 2025 Google LLC
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

from pathlib import Path
import sys

from scripts import run_and_publish_benchmark

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))


def test_collect_benchmark_result(tmp_path):
    # Create dummy log files
    (tmp_path / "benchmark1.bytesprocessed").write_text("100")
    (tmp_path / "benchmark1.slotmillis").write_text("1000")
    (tmp_path / "benchmark1.bq_exec_time_seconds").write_text("1.0")
    (tmp_path / "benchmark1.local_exec_time_seconds").write_text("2.0")
    (tmp_path / "benchmark1.query_char_count").write_text("50")
    (tmp_path / "benchmark1.totalrows").write_text("10")

    # Collect the benchmark results
    df, error_message = run_and_publish_benchmark.collect_benchmark_result(
        str(tmp_path), 1
    )

    # Assert that the DataFrame is correct
    assert error_message is None
    assert len(df) == 1
    assert df["Benchmark_Name"][0] == "benchmark1"
    assert df["Bytes_Processed"][0] == 100
    assert df["Slot_Millis"][0] == 1000
    assert df["BigQuery_Execution_Time_Sec"][0] == 1.0
    assert df["Local_Execution_Time_Sec"][0] == 2.0
    assert df["Query_Char_Count"][0] == 50
    assert df["Total_Rows"][0] == 10
