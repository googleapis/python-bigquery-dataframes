# Copyright 2024 Google LLC
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

import argparse
import datetime
import json
import os
import pathlib
from pathlib import Path
import subprocess
import sys
from typing import Dict, List, Union

import pandas as pd
import pandas_gbq

LOGGING_NAME_ENV_VAR = "BIGFRAMES_PERFORMANCE_LOG_NAME"
CURRENT_DIRECTORY = pathlib.Path(__file__).parent.absolute()


def benchmark_process(args, log_env_name_var, filename=None, region=None):
    env = os.environ.copy()
    current_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        os.path.join(os.getcwd(), "tests") + os.pathsep + current_pythonpath
    )

    if region:
        env["BIGQUERY_LOCATION"] = region
    env[LOGGING_NAME_ENV_VAR] = log_env_name_var
    result = subprocess.run(
        args, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    print(result.stdout)
    if result.returncode != 0:
        print(result.stderr)


def collect_benchmark_result(benchmark_path: str) -> pd.DataFrame:
    """Generate a DataFrame report on HTTP queries, bytes processed, slot time and execution time from log files."""
    path = Path(benchmark_path)
    try:
        results_dict: Dict[str, List[Union[int, float, None]]] = {}
        bytes_files = sorted(path.rglob("*.bytesprocessed"))
        millis_files = sorted(path.rglob("*.slotmillis"))
        bq_seconds_files = sorted(path.rglob("*.bq_exec_time_seconds"))

        local_seconds_files = sorted(path.rglob("*.local_exec_time_seconds"))
        has_local_seconds = len(local_seconds_files) > 0

        if has_local_seconds:
            if not (
                len(bytes_files)
                == len(millis_files)
                == len(local_seconds_files)
                == len(bq_seconds_files)
            ):
                raise ValueError(
                    "Mismatch in the number of report files for bytes, millis, and seconds."
                )
        else:
            if not (len(bytes_files) == len(millis_files) == len(bq_seconds_files)):
                raise ValueError(
                    "Mismatch in the number of report files for bytes, millis, and seconds."
                )

        for idx in range(len(bytes_files)):
            bytes_file = bytes_files[idx]
            millis_file = millis_files[idx]
            bq_seconds_file = bq_seconds_files[idx]
            filename = bytes_file.relative_to(path).with_suffix("")

            if filename != millis_file.relative_to(path).with_suffix(
                ""
            ) or filename != bq_seconds_file.relative_to(path).with_suffix(""):
                raise ValueError(
                    "File name mismatch among bytes, millis, and seconds reports."
                )

            if has_local_seconds:
                local_seconds_file = local_seconds_files[idx]
                if filename != local_seconds_file.relative_to(path).with_suffix(""):
                    raise ValueError(
                        "File name mismatch among bytes, millis, and seconds reports."
                    )

            with open(bytes_file, "r") as file:
                lines = file.read().splitlines()
                query_count = len(lines)
                total_bytes = sum(int(line) for line in lines)

            with open(millis_file, "r") as file:
                lines = file.read().splitlines()
                total_slot_millis = sum(int(line) for line in lines)

            if has_local_seconds:
                with open(local_seconds_file, "r") as file:
                    local_seconds = float(file.readline().strip())
            else:
                local_seconds = None

            with open(bq_seconds_file, "r") as file:
                lines = file.read().splitlines()
                bq_seconds = sum(float(line) for line in lines)

            results_dict[str(filename)] = [
                query_count,
                total_bytes,
                total_slot_millis,
                local_seconds,
                bq_seconds,
            ]
    finally:
        for files_to_remove in (
            path.rglob("*.bytesprocessed"),
            path.rglob("*.slotmillis"),
            path.rglob("*.local_exec_time_seconds"),
            path.rglob("*.bq_exec_time_seconds"),
        ):
            for log_file in files_to_remove:
                log_file.unlink()

    columns = [
        "Query_Count",
        "Bytes_Processed",
        "Slot_Millis",
        "Local_Execution_Time_Sec",
        "BigQuery_Execution_Time_Sec",
    ]

    benchmark_metrics = pd.DataFrame.from_dict(
        results_dict,
        orient="index",
        columns=columns,
    )

    print("---BIGQUERY USAGE REPORT---")
    for index, row in benchmark_metrics.iterrows():
        print(
            f"{index} - query count: {row['Query_Count']},"
            f" bytes processed sum: {row['Bytes_Processed']},"
            f" slot millis sum: {row['Slot_Millis']},"
            f" local execution time: {row['Local_Execution_Time_Sec']} seconds,"
            f" bigquery execution time: {row['BigQuery_Execution_Time_Sec']} seconds"
        )

    cumulative_queries = benchmark_metrics["Query_Count"].sum()
    cumulative_bytes = benchmark_metrics["Bytes_Processed"].sum()
    cumulative_slot_millis = benchmark_metrics["Slot_Millis"].sum()
    cumulative_local_seconds = benchmark_metrics["Local_Execution_Time_Sec"].sum(
        skipna=True
    )
    cumulative_bq_seconds = benchmark_metrics["BigQuery_Execution_Time_Sec"].sum()

    print(
        f"---total queries: {cumulative_queries}, "
        f"total bytes: {cumulative_bytes}, "
        f"total slot millis: {cumulative_slot_millis}, "
        f"Total local execution time: {cumulative_local_seconds} seconds, "
        f"Total bigquery execution time: {cumulative_bq_seconds} seconds---"
    )

    return benchmark_metrics.reset_index().rename(columns={"index": "Benchmark_Name"})


def get_repository_status():
    if "KOKORO_JOB_NAME" in os.environ:
        current_directory = os.getcwd()
        subprocess.run(
            ["git", "config", "--global", "--add", "safe.directory", current_directory],
            check=True,
        )

    git_hash = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], text=True
    ).strip()
    bigframes_version = subprocess.check_output(
        ["python", "-c", "import bigframes; print(bigframes.__version__)"], text=True
    ).strip()
    release_version = (
        f"{bigframes_version}dev{datetime.datetime.now().strftime('%Y%m%d')}+{git_hash}"
    )

    return {
        "benchmark_start_time": datetime.datetime.now().isoformat(),
        "git_hash": git_hash,
        "bigframes_version": bigframes_version,
        "release_version": release_version,
        "python_version": sys.version,
    }


def find_config(start_path):
    target_file = "config.jsonl"
    current_path = Path(start_path).resolve()
    if current_path.is_file():
        current_path = current_path.parent

    levels_checked = 0
    while current_path.name != "benchmark" and levels_checked < 3:
        config_path = current_path / target_file
        if config_path.exists():
            return config_path
        if current_path.parent == current_path:
            break
        current_path = current_path.parent
        levels_checked += 1

    return None


def process_benchmark(benchmark: str):
    print(benchmark)
    config_path = find_config(benchmark)

    if config_path:
        benchmark_configs = []
        with open(config_path, "r") as f:
            for line in f:
                config = json.loads(line)
                python_args = [f"--{key}={value}" for key, value in config.items()]
                suffix = (
                    config["benchmark_suffix"]
                    if "benchmark_suffix" in config
                    else "_".join(f"{key}_{value}" for key, value in config.items())
                )
                benchmark_configs.append((suffix, python_args))
    else:
        benchmark_configs = [(None, [])]

    for benchmark_config in benchmark_configs:
        args = ["python", str(benchmark)]
        args.extend(benchmark_config[1])
        log_env_name_var = str(benchmark)
        if benchmark_config[0] is not None:
            log_env_name_var += f"_{benchmark_config[0]}"
        benchmark_process(args=args, log_env_name_var=log_env_name_var)


def process_notebook_benchmark(benchmark_file: str, region: str):
    export_file = f"{benchmark_file}_{region}" if region else benchmark_file
    log_env_name_var = os.path.basename(export_file)
    # TODO(shobs): For some reason --retries arg masks exceptions occurred in
    # notebook failures, and shows unhelpful INTERNALERROR. Investigate that
    # and enable retries if we can find a way to surface the real exception
    # bacause the notebook is running against real GCP and something may fail
    # due to transient issues.
    pytest_command = [
        "py.test",
        "--nbmake",
        "--nbmake-timeout=900",  # 15 minutes
    ]
    benchmark_args = (*pytest_command, benchmark_file)

    benchmark_process(
        args=benchmark_args,
        log_env_name_var=log_env_name_var,
        filename=export_file,
        region=region,
    )


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run benchmarks for different scenarios."
    )
    parser.add_argument(
        "--notebook",
        action="store_true",
        help="Set this flag to run the benchmark as a notebook. If not set, it assumes a Python (.py) file.",
    )

    parser.add_argument(
        "--benchmark-path",
        type=str,
        default=None,
        help="Specify the file path to the benchmark script, either a Jupyter notebook or a Python script.",
    )

    parser.add_argument(
        "--region",
        type=str,
        default=None,
        help="Specify the region where the benchmark will be executed or where the data resides. This parameter is optional.",
    )

    parser.add_argument(
        "--publish-benchmarks",
        type=str,
        default=None,
        help="Set the benchmarks to be published to BigQuery.",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    if args.publish_benchmarks:
        bigquery_table = (
            "bigframes-metrics.benchmark_report.notebook_benchmark"
            if args.notebook
            else "bigframes-metrics.benchmark_report.benchmark"
        )
        benchmark_metrics = collect_benchmark_result(args.publish_benchmarks)
        repo_status = get_repository_status()

        for idx, col in enumerate(repo_status.keys()):
            benchmark_metrics.insert(idx, col, repo_status[col])

        if os.getenv("NOX_SESSION", "local") in (
            "benchmark-load",
            "notebook-continuous",
        ):
            pandas_gbq.to_gbq(
                dataframe=benchmark_metrics,
                destination_table=bigquery_table,
                if_exists="append",
            )
    elif args.notebook:
        process_notebook_benchmark(args.benchmark_path, args.region)
    else:
        process_benchmark(args.benchmark_path)


if __name__ == "__main__":
    main()
